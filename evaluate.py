"""Compare the neural network's upscaled result against classical resize
baselines (nearest/bilinear/bicubic/Lanczos/sharpened-bicubic) using PSNR
and SSIM, and write a labeled side-by-side diagnostic panel."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageFilter

from metrics import compute_color_metrics, compute_edge_metrics, compute_luma_metrics, compute_metrics
from panel import make_panel

SCRIPT_DIR = Path(__file__).resolve().parent

RESAMPLE_METHODS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def resize(input_img, size, method):
    return np.array(Image.fromarray(input_img).resize(size, resample=RESAMPLE_METHODS[method]))


def sharpened_bicubic(input_img, size):
    im = Image.fromarray(input_img).resize(size, resample=Image.BICUBIC)
    im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
    return np.array(im)


def crop_to_match(img, offset, outh, outw):
    """Crop a full-size image to the interior region the network actually
    predicts (it skips a 1-input-pixel border, i.e. `scale - 1` output
    pixels on each side)."""
    return img[offset:offset + outh, offset:offset + outw]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the neural network result against classical resize baselines."
    )
    parser.add_argument("--input", default=str(SCRIPT_DIR / "inkami.png"), help="low quality source image")
    parser.add_argument("--target", default=str(SCRIPT_DIR / "outkami.png"), help="clean high resolution reference image")
    parser.add_argument("--neural", default="result.png", help="the neural network's upscaled output (from main.py)")
    parser.add_argument("--scale", type=int, default=3, help="upscale factor used to train the network")
    parser.add_argument("--panel", default="comparison.png", help="where to write the side-by-side diagnostic panel")
    parser.add_argument("--metrics", default="metrics.json", help="where to write the PSNR/SSIM metrics as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    input_img = imageio.imread(args.input)[:, :, :3]
    target_img = imageio.imread(args.target)[:, :, :3]
    neural_img = imageio.imread(args.neural)[:, :, :3]

    outh, outw = neural_img.shape[:2]
    offset = args.scale - 1
    target_crop = crop_to_match(target_img, offset, outh, outw)

    size = (target_img.shape[1], target_img.shape[0])  # PIL wants (width, height)
    candidates = {name: resize(input_img, size, name) for name in RESAMPLE_METHODS}
    candidates["sharpened_bicubic"] = sharpened_bicubic(input_img, size)
    candidates = {name: crop_to_match(img, offset, outh, outw) for name, img in candidates.items()}
    candidates["neural"] = neural_img

    metrics = {}
    for name, img in candidates.items():
        m = compute_metrics(img, target_crop)
        m.update(compute_edge_metrics(img, target_crop))
        m.update(compute_luma_metrics(img, target_crop))
        m.update(compute_color_metrics(img, target_crop))
        metrics[name] = m

    print(f"{'method':<18}{'psnr':>8}{'ssim':>8}{'edge_psnr':>11}{'edge_ssim':>11}{'luma_psnr':>11}{'delta_e':>9}")
    for name, m in sorted(metrics.items(), key=lambda kv: -kv[1]["edge_psnr"]):
        print(
            f"{name:<18}{m['psnr']:>8.2f}{m['ssim']:>8.4f}{m['edge_psnr']:>11.2f}"
            f"{m['edge_ssim']:>11.4f}{m['luma_psnr']:>11.2f}{m['delta_e_mean']:>9.2f}"
        )

    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    panel_images = [("input", input_img), ("target", target_crop)] + [
        (name, img) for name, img in candidates.items()
    ]
    make_panel(panel_images, args.panel, cols=3)
    print(f"\nMetrics written to {args.metrics}")
    print(f"Comparison panel written to {args.panel}")


if __name__ == "__main__":
    main()
