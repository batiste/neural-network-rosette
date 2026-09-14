"""Preview sheet generator: samples random crops from the source images and
runs them through each moire preset so the degradation parameters can be
visually inspected before training."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from moire import PRESETS, degrade
from panel import make_panel

SCRIPT_DIR = Path(__file__).resolve().parent


def load_sources(source_dir):
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in Path(source_dir).iterdir() if p.suffix.lower() in exts)


def random_hr_crop(image_path, hr_size, rng):
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    if w < hr_size or h < hr_size:
        im = im.resize((max(w, hr_size), max(h, hr_size)))
        w, h = im.size
    x = int(rng.integers(0, w - hr_size + 1))
    y = int(rng.integers(0, h - hr_size + 1))
    return np.array(im.crop((x, y, x + hr_size, y + hr_size)))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a preview sheet of synthetic moire presets.")
    parser.add_argument("--sources", default=str(SCRIPT_DIR / "sources"), help="directory of clean source images")
    parser.add_argument("--out", default="moire_preview.png", help="where to write the preview sheet")
    parser.add_argument("--hr-size", type=int, default=300, help="high-res crop size in pixels (default: %(default)s)")
    parser.add_argument("--scale", type=int, default=3, help="downscale factor (default: %(default)s)")
    parser.add_argument("--samples", type=int, default=4, help="number of source crops to preview (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    sources = load_sources(args.sources)
    if not sources:
        raise SystemExit(f"No source images found in {args.sources}")

    panel_images = []
    for _ in range(args.samples):
        path = sources[int(rng.integers(0, len(sources)))]
        hr = random_hr_crop(path, args.hr_size, rng)
        panel_images.append((f"clean ({path.name})", hr))
        for preset in PRESETS:
            lr = degrade(hr, args.scale, rng, preset=preset)
            lr_upscaled = np.array(
                Image.fromarray(lr).resize((hr.shape[1], hr.shape[0]), resample=Image.NEAREST)
            )
            panel_images.append((preset, lr_upscaled))

    make_panel(panel_images, args.out, cols=len(PRESETS) + 1)
    print(f"Preview sheet written to {args.out}")


if __name__ == "__main__":
    main()
