"""Real-world validation: a strong classical upscale vs. network output,
for actual card photos (bear.webp, lotus.webp), not synthetic crops --
the honest test of whether this generalizes beyond the training
distribution.

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
        description="Build a full-resolution, lossless source/classical/network comparison for real card photos."
    )
    parser.add_argument("--pairs", nargs="+", default=["bear", "lotus"],
                         help="basenames with a <name>.webp source and <name>_upscaled.png output")
    parser.add_argument("--out", default="real_world_results.png")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for name in args.pairs:
        source = Image.open(SCRIPT_DIR / f"{name}.webp").convert("RGB")
        output = Image.open(SCRIPT_DIR / f"{name}_upscaled.png").convert("RGB")
        scale = round(output.width / source.width)
        # Lanczos: the strongest classical resampling filter (best PSNR/SSIM
        # of nearest/bilinear/bicubic/lanczos/sharpened-bicubic in evaluate.py)
        classical = source.resize((source.width * scale, source.height * scale), resample=Image.LANCZOS)
        rows.append((name, [
            (f"{scale}x lanczos", classical),
            ("network output", output),
        ]))

    n_cols = len(rows[0][1])
    col_widths = [max(cells[c][1].width for _, cells in rows) for c in range(n_cols)]
    row_heights = [max(img.height for _, img in cells) + LABEL_H for _, cells in rows]

    panel_w = PAD * (n_cols + 1) + sum(col_widths)
    panel_h = PAD * (len(rows) + 1) + sum(row_heights)
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    y = PAD
    for (name, cells), row_h in zip(rows, row_heights):
        x = PAD
        for c, (label, img) in enumerate(cells):
            panel.paste(img, (x, y))
            draw.text((x, y + img.height + 4), f"{name}: {label}", fill="black", font=font)
            x += col_widths[c] + PAD
        y += row_h + PAD

    panel.save(args.out)
    print(f"Wrote {args.out} ({panel.width}x{panel.height})")


if __name__ == "__main__":
    main()
