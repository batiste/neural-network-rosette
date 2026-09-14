"""Real-world validation: source vs. network output for actual card
photos (bear.webp, lotus.webp), not synthetic crops -- the honest test
of whether this generalizes beyond the training distribution."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from panel import make_panel

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Build a source/output comparison panel for real card photos.")
    parser.add_argument("--pairs", nargs="+", default=["bear", "lotus"],
                         help="basenames with a <name>.webp source and <name>_upscaled.png output")
    parser.add_argument("--out", default="real_world_results.png")
    parser.add_argument("--cell-size", type=int, default=380)
    return parser.parse_args()


def main():
    args = parse_args()
    panel_images = []
    for name in args.pairs:
        source_path = SCRIPT_DIR / f"{name}.webp"
        output_path = SCRIPT_DIR / f"{name}_upscaled.png"
        source = np.array(Image.open(source_path).convert("RGB"))
        output = np.array(Image.open(output_path).convert("RGB"))
        panel_images.append((f"{name}: source", source))
        panel_images.append((f"{name}: network output", output))

    make_panel(panel_images, args.out, cols=2, cell_size=args.cell_size)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
