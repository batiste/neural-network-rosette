"""The actual "does this work" demo: for a few random crops of the source
images, show the clean source, a synthetic moire-degraded version, and what
the trained network produces from that degraded input."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import list_sources
from device import get_device
from graphics import add_synthetic_graphics
from infer import load_model
from moire import PRESETS, degrade
from panel import make_panel

SCRIPT_DIR = Path(__file__).resolve().parent
PRESET_NAMES = list(PRESETS)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a source / moire / network-output demo panel.")
    parser.add_argument("--sources", default=str(SCRIPT_DIR / "sources"))
    parser.add_argument("--checkpoint", default=str(SCRIPT_DIR / "checkpoints" / "best.pt"))
    parser.add_argument("--out", default="results.png")
    parser.add_argument("--hr-size", type=int, default=300)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default=None, help="fixed preset name, otherwise randomized per sample")
    parser.add_argument("--graphics-prob", type=float, default=0.5,
                         help="probability of overlaying synthetic text/lines onto a sample before degrading it")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    model, scale = load_model(args.checkpoint, device)
    if args.hr_size % scale != 0:
        raise SystemExit(f"--hr-size must be divisible by the checkpoint's scale ({scale})")

    sources = list_sources(args.sources)
    if not sources:
        raise SystemExit(f"No source images found in {args.sources}")

    panel_images = []
    for _ in range(args.samples):
        path = sources[int(rng.integers(0, len(sources)))]
        im = Image.open(path).convert("RGB")
        w, h = im.size
        x = int(rng.integers(0, w - args.hr_size + 1))
        y = int(rng.integers(0, h - args.hr_size + 1))
        hr = np.array(im.crop((x, y, x + args.hr_size, y + args.hr_size)))
        protect_mask = None
        if rng.random() < args.graphics_prob:
            hr, protect_mask = add_synthetic_graphics(hr, rng)

        preset = args.preset or PRESET_NAMES[int(rng.integers(0, len(PRESET_NAMES)))]
        lr = degrade(hr, scale, rng, preset=preset, protect_mask=protect_mask)

        with torch.no_grad():
            lr_t = torch.from_numpy(lr.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
            pred = model(lr_t)[0].clamp(0, 1)
        pred_img = (pred * 255).round().byte().permute(1, 2, 0).cpu().numpy()

        lr_upscaled = np.array(Image.fromarray(lr).resize((args.hr_size, args.hr_size), resample=Image.NEAREST))

        panel_images.append((f"source ({path.name})", hr))
        panel_images.append((f"moire ({preset})", lr_upscaled))
        panel_images.append(("network output", pred_img))

    make_panel(panel_images, args.out, cols=3, cell_size=260)
    print(f"Results sheet written to {args.out}")


if __name__ == "__main__":
    main()
