"""Real-world validation: source vs. network output for actual card
photos (bear.webp, lotus.webp), not synthetic crops -- the honest test
of whether this generalizes beyond the training distribution.

Composited at each image's full native resolution with zero resampling
(not panel.py's make_panel, which thumbnails everything down to a small
fixed cell size -- exactly the kind of downscaling that would hide the
sharpness difference this comparison exists to show)."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PAD = 24
LABEL_H = 36


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a full-resolution, lossless source/output comparison for real card photos."
    )
    parser.add_argument("--pairs", nargs="+", default=["bear", "lotus"],
                         help="basenames with a <name>.webp source and <name>_upscaled.png output")
    parser.add_argument("--out", default="real_world_results.png")
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = []
    for name in args.pairs:
        source = Image.open(SCRIPT_DIR / f"{name}.webp").convert("RGB")
        output = Image.open(SCRIPT_DIR / f"{name}_upscaled.png").convert("RGB")
        pairs.append((name, source, output))

    source_col_w = max(s.width for _, s, o in pairs)
    output_col_w = max(o.width for _, s, o in pairs)
    row_heights = [max(s.height, o.height) + LABEL_H for _, s, o in pairs]

    panel_w = PAD * 3 + source_col_w + output_col_w
    panel_h = PAD * (len(pairs) + 1) + sum(row_heights)
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    y = PAD
    for (name, source, output), row_h in zip(pairs, row_heights):
        panel.paste(source, (PAD, y))
        panel.paste(output, (PAD * 2 + source_col_w, y))
        draw.text((PAD, y + source.height + 4), f"{name}: source", fill="black", font=font)
        draw.text((PAD * 2 + source_col_w, y + output.height + 4), f"{name}: network output", fill="black", font=font)
        y += row_h + PAD

    panel.save(args.out)
    print(f"Wrote {args.out} ({panel.width}x{panel.height})")


if __name__ == "__main__":
    main()
