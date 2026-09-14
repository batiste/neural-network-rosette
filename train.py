"""Train the TinySRNet CNN to jointly upscale and demoire images, using
synthetically degraded patches from the source images (see moire.py)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from dataset import MoireDataset, list_sources, split_sources
from device import get_device
from model import TinySRNet
from panel import make_panel

SCRIPT_DIR = Path(__file__).resolve().parent

_SOBEL_X = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3)


def edge_loss(pred, target):
    channels = pred.shape[1]
    kx = _SOBEL_X.to(pred.device, pred.dtype).repeat(channels, 1, 1, 1)
    ky = _SOBEL_Y.to(pred.device, pred.dtype).repeat(channels, 1, 1, 1)

    def gradient_magnitude(img):
        gx = F.conv2d(img, kx, padding=1, groups=channels)
        gy = F.conv2d(img, ky, padding=1, groups=channels)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    return F.l1_loss(gradient_magnitude(pred), gradient_magnitude(target))


def to_uint8_image(t):
    return (t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnrs, ssims = [], []
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        for p, t in zip(pred, hr):
            p_img, t_img = to_uint8_image(p), to_uint8_image(t)
            psnrs.append(peak_signal_noise_ratio(t_img, p_img, data_range=255))
            ssims.append(structural_similarity(t_img, p_img, channel_axis=2, data_range=255))
    return float(np.mean(psnrs)), float(np.mean(ssims))


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
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--edge-weight", type=float, default=0.5)
    parser.add_argument("--val-count", type=int, default=2, help="number of source images held out for validation")
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
    )
    val_dataset = MoireDataset(
        val_paths, args.hr_size, args.scale,
        epoch_length=args.val_samples, deterministic=True, seed=1234,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    model = TinySRNet(scale=args.scale, channels=args.channels, num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_psnr = -float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", best_psnr)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            optimizer.zero_grad()
            pred = model(lr_batch)
            loss = F.l1_loss(pred, hr_batch) + args.edge_weight * edge_loss(pred, hr_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * lr_batch.shape[0]

        train_loss = running_loss / len(train_dataset)
        val_psnr, val_ssim = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={train_loss:.4f}  val_psnr={val_psnr:.2f}  val_ssim={val_ssim:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_psnr": val_psnr, "val_ssim": val_ssim})

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "scale": args.scale,
            "channels": args.channels,
            "num_blocks": args.num_blocks,
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint["best_psnr"] = best_psnr
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        if (epoch + 1) % args.preview_every == 0 or epoch == args.epochs - 1:
            write_preview(model, val_dataset, device, checkpoint_dir / f"preview_epoch_{epoch + 1:03d}.png")

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
