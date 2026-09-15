"""Real-world validation: one compact comparison image per card photo
(bear.webp, lotus.webp), not synthetic crops -- the honest test of
whether this generalizes beyond the training distribution.

Each output image has two rows, both at the same full width:
  1. Both cards at the network output's full native size (the source is
     nearest-neighbor upscaled to match, honestly showing its actual
     blockiness rather than hiding it behind a small thumbnail).
  2. A 3x-zoomed crop of a representative region (bottom of the art, the
     type line, and the start of the rules text) so the actual
     pixel-level sharpness difference is visible.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PAD = 16
LABEL_H = 56
LABEL_FONT_SIZE = 40

# Crop region in SOURCE (native) pixel coordinates: bottom of the art,
# the type line ("Summon Bears" / "Mono Artifact"), and the start of the
# rules text box.
MID_CROP = (0, 380, 672, 620)

_LABEL_FONT_CANDIDATES = ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc")


def _label_font(size=LABEL_FONT_SIZE):
    for path in _LABEL_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def labeled_row(cells, pad=PAD, label_h=LABEL_H):
    """cells: [(label, PIL.Image), ...] laid out left to right."""
    row_height = max(img.height for _, img in cells) + label_h
    row_width = sum(img.width for _, img in cells) + pad * (len(cells) + 1)
    row = Image.new("RGB", (row_width, row_height + pad * 2), "white")
    draw = ImageDraw.Draw(row)
    font = _label_font()
    x = pad
    for label, img in cells:
        row.paste(img, (x, pad))
        draw.text((x, pad + img.height + 8), label, fill="black", font=font)
        x += img.width + pad
    return row


def stack_rows(rows, pad=PAD):
    width = max(r.width for r in rows)
    height = sum(r.height for r in rows) + pad * (len(rows) + 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = pad
    for r in rows:
        canvas.paste(r, ((width - r.width) // 2, y))
        y += r.height + pad
    return canvas


def build_comparison(name):
    source = Image.open(SCRIPT_DIR / f"{name}.webp").convert("RGB")
    output = Image.open(SCRIPT_DIR / f"{name}_upscaled.png").convert("RGB")
    scale = round(output.width / source.width)

    # Row 1: both cards at the network output's full native size --
    # nearest-neighbor upscale the source to match, honestly showing its
    # actual low-res blockiness rather than smoothing it away.
    source_big = source.resize(output.size, resample=Image.NEAREST)
    row1 = labeled_row([("source", source_big), ("network output", output)])

    # Row 2: the same physical region, zoomed. The source crop is
    # upscaled with nearest-neighbor (honest -- shows the actual pixel
    # blockiness rather than a smooth resample hiding it); the output
    # crop is just the corresponding region at its native full detail.
    x0, y0, x1, y1 = MID_CROP
    source_crop = source.crop((x0, y0, x1, y1))
    source_crop_zoomed = source_crop.resize(
        (source_crop.width * scale, source_crop.height * scale), resample=Image.NEAREST
    )
    output_crop = output.crop((x0 * scale, y0 * scale, x1 * scale, y1 * scale))
    row2 = labeled_row([
        (f"source ({scale}x zoom, nearest)", source_crop_zoomed),
        ("network output (zoom)", output_crop),
    ])

    return stack_rows([row1, row2])


def parse_args():
    parser = argparse.ArgumentParser(description="Build a compact per-card real-world comparison image.")
    parser.add_argument("--names", nargs="+", default=["bear", "lotus"],
                         help="basenames with a <name>.webp source and <name>_upscaled.png output")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in args.names:
        panel = build_comparison(name)
        out_path = SCRIPT_DIR / f"{name}_comparison.png"
        panel.save(out_path)
        print(f"Wrote {out_path} ({panel.width}x{panel.height})")


if __name__ == "__main__":
    main()
