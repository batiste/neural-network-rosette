"""Train the TinySRNet CNN to jointly upscale and demoire images, using
synthetically degraded patches from the source images (see moire.py)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MoireDataset, list_sources, split_sources
from device import get_device
from losses import laplacian_pyramid_loss
from metrics import compute_color_metrics, compute_edge_metrics, compute_luma_metrics, compute_metrics
from model import TinySRNet
from panel import make_panel

SCRIPT_DIR = Path(__file__).resolve().parent


def to_uint8_image(t):
    return (t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def evaluate(model, loader, device):
    """Whole-image PSNR/SSIM, the edge-masked variant (fidelity on
    text/line/hard-edge pixels specifically), luma PSNR/SSIM (brightness
    alone -- what human vision is most sensitive to), and mean Lab
    Delta-E (perceptual color error). See metrics.py."""
    model.eval()
    keys = ["psnr", "ssim", "edge_psnr", "edge_ssim", "luma_psnr", "luma_ssim", "delta_e_mean"]
    values = {k: [] for k in keys}
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        for p, t in zip(pred, hr):
            p_img, t_img = to_uint8_image(p), to_uint8_image(t)
            for k, v in compute_metrics(p_img, t_img).items():
                values[k].append(v)
            em = compute_edge_metrics(p_img, t_img)
            if em["edge_fraction"] > 0:
                values["edge_psnr"].append(em["edge_psnr"])
                values["edge_ssim"].append(em["edge_ssim"])
            for k, v in compute_luma_metrics(p_img, t_img).items():
                values[k].append(v)
            values["delta_e_mean"].append(compute_color_metrics(p_img, t_img)["delta_e_mean"])
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in values.items()}


@torch.no_grad()
def write_preview(model, val_dataset, device, filename, n=3):
    model.eval()
    panel_images = []
    for i in range(min(n, len(val_dataset))):
        lr, hr = val_dataset[i]
        pred = model(lr.unsqueeze(0).to(device))[0]
        bicubic = F.interpolate(lr.unsqueeze(0), scale_factor=val_dataset.scale, mode="bicubic", align_corners=False)[0]
        panel_images.append((f"input {i}", to_uint8_image(lr)))
        panel_images.append((f"bicubic {i}", to_uint8_image(bicubic)))
        panel_images.append((f"prediction {i}", to_uint8_image(pred)))
        panel_images.append((f"target {i}", to_uint8_image(hr)))
    make_panel(panel_images, filename, cols=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CNN demoire/upscale model.")
    parser.add_argument("--sources", default=str(SCRIPT_DIR / "sources"))
    parser.add_argument("--checkpoint-dir", default=str(SCRIPT_DIR / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hr-size", type=int, default=192, help="crop size in the clean image (must be divisible by --scale)")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--channels", type=int, default=80)
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--edge-weight", type=float, default=3.0,
                         help="weight on the Laplacian-pyramid (multi-scale frequency) loss term")
    parser.add_argument("--graphics-prob", type=float, default=0.5,
                         help="probability of overlaying synthetic text/lines onto a training crop before degrading it")
    parser.add_argument("--synthetic-share", type=float, default=0.3,
                         help="fraction of sampling probability mass given to generate_synthetic_sources.py "
                              "images as a group, independent of file count or size (default: %(default)s)")
    parser.add_argument("--val-count", type=int, default=4, help="number of source images held out for validation")
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hr_size % args.scale != 0:
        raise SystemExit("--hr-size must be divisible by --scale")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    device = args.device and torch.device(args.device) or get_device()
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sources = list_sources(args.sources)
    train_paths, val_paths = split_sources(sources, val_count=args.val_count)
    print(f"Training on {len(train_paths)} images, validating on {[p.name for p in val_paths]}")

    train_dataset = MoireDataset(
        train_paths, args.hr_size, args.scale,
        epoch_length=args.steps_per_epoch * args.batch_size,
        graphics_prob=args.graphics_prob, synthetic_share=args.synthetic_share,
    )
    val_dataset = MoireDataset(
        val_paths, args.hr_size, args.scale,
        epoch_length=args.val_samples, deterministic=True, seed=1234,
        graphics_prob=args.graphics_prob, synthetic_share=args.synthetic_share,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    model = TinySRNet(scale=args.scale, channels=args.channels, num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    # Select "best" by edge-region PSNR, not whole-image PSNR: whole-image
    # scores are dominated by large flat/textured areas and reward blur,
    # while edge fidelity (text, line art, hard boundaries) is what we
    # actually care about here.
    best_edge_psnr = -float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_edge_psnr = ckpt.get("best_edge_psnr", best_edge_psnr)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            optimizer.zero_grad()
            pred = model(lr_batch)
            loss = F.l1_loss(pred, hr_batch) + args.edge_weight * laplacian_pyramid_loss(pred, hr_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * lr_batch.shape[0]

        train_loss = running_loss / len(train_dataset)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1}/{args.epochs}  loss={train_loss:.4f}  "
            f"val_psnr={val_metrics['psnr']:.2f}  val_ssim={val_metrics['ssim']:.4f}  "
            f"edge_psnr={val_metrics['edge_psnr']:.2f}  edge_ssim={val_metrics['edge_ssim']:.4f}  "
            f"luma_psnr={val_metrics['luma_psnr']:.2f}  delta_e={val_metrics['delta_e_mean']:.2f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_edge_psnr": best_edge_psnr,
            "scale": args.scale,
            "channels": args.channels,
            "num_blocks": args.num_blocks,
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        if val_metrics["edge_psnr"] > best_edge_psnr:
            best_edge_psnr = val_metrics["edge_psnr"]
            checkpoint["best_edge_psnr"] = best_edge_psnr
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        if (epoch + 1) % args.preview_every == 0 or epoch == args.epochs - 1:
            write_preview(model, val_dataset, device, checkpoint_dir / f"preview_epoch_{epoch + 1:03d}.png")

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
