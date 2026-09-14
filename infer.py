"""Run a trained TinySRNet checkpoint on an arbitrary image, tiling with
overlap so large scans don't have to fit in memory (or on the MPS GPU) at
once and don't show seams at tile boundaries."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from device import get_device
from model import TinySRNet

SCRIPT_DIR = Path(__file__).resolve().parent


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TinySRNet(scale=ckpt["scale"], channels=ckpt["channels"], num_blocks=ckpt["num_blocks"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["scale"]


@torch.no_grad()
def run_tiled(model, img_tensor, scale, tile_size, context, device):
    """Tiles are moved to `device` one at a time so only a single tile's
    activations sit on the GPU/MPS device at once; the output accumulates
    on the CPU so a large image's full-resolution buffer doesn't also have
    to fit in GPU memory."""
    _, _, h, w = img_tensor.shape
    padded = F.pad(img_tensor, (context, context, context, context), mode="reflect")
    out = torch.zeros(1, 3, h * scale, w * scale)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            th = min(tile_size, h - y)
            tw = min(tile_size, w - x)
            tile = padded[:, :, y:y + th + 2 * context, x:x + tw + 2 * context]
            pred = model(tile.to(device)).cpu()
            core = pred[:, :, context * scale:(context + th) * scale, context * scale:(context + tw) * scale]
            out[:, :, y * scale:(y + th) * scale, x * scale:(x + tw) * scale] = core

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained TinySRNet checkpoint on an image.")
    parser.add_argument("--checkpoint", default=str(SCRIPT_DIR / "checkpoints" / "best.pt"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="upscaled.png")
    parser.add_argument("--tile-size", type=int, default=256, help="LR tile size in pixels for large images")
    parser.add_argument("--context", type=int, default=16, help="extra context pixels around each tile to avoid seams")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()
    model, scale = load_model(args.checkpoint, device)

    img = np.array(Image.open(args.input).convert("RGB")).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    out = run_tiled(model, img_tensor, scale, args.tile_size, args.context, device)
    out_img = (out[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(out_img).save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
