"""Generate synthetic graphic-design source images -- dense text, gradients,
lines, and sharp shapes -- and drop them into sources/.

The photographic/painterly art sources give the model plenty of soft
texture but essentially no graphic-design content to train on directly
(graphics.py's on-the-fly overlays only touch a small fraction of a
crop). These images are graphic-design content wholesale, so the model
sees a lot more of it, more densely, than sparse overlays alone provide.
Regenerate anytime with a different --seed for a fresh batch; nothing here
is hand-crafted, so there's no reason to commit the output.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from graphics import _contrasting_color, _luminance, _random_font, _random_text, offset_color

SCRIPT_DIR = Path(__file__).resolve().parent
SIZE = (1000, 1400)


def _random_bg(rng, light):
    lo, hi = (200, 256) if light else (0, 55)
    return tuple(int(c) for c in rng.integers(lo, hi, size=3))


def gen_text_block(rng, light_bg):
    """Paragraphs of text, contrast-forced against a solid background."""
    w, h = SIZE
    bg = _random_bg(rng, light=light_bg)
    im = Image.new("RGB", SIZE, bg)
    draw = ImageDraw.Draw(im)
    bg_luminance = _luminance(bg)

    y = int(h * 0.03)
    margin = int(w * 0.05)
    while y < h - margin:
        font_size = int(rng.integers(9, 80))
        font = _random_font(rng, font_size)
        text = _random_text(rng, min_len=10, max_len=45)
        color = _contrasting_color(rng, bg_luminance)
        draw.text((margin, y), text, font=font, fill=color)
        y += font_size + int(rng.integers(6, 18))
    return np.array(im)


def _noise_octave(rng, size, cell):
    """One octave of value noise: low-res random values smoothly
    upsampled to full size."""
    w, h = size
    small = rng.uniform(-1, 1, size=(max(2, h // cell), max(2, w // cell))).astype(np.float32)
    im = Image.fromarray(((small + 1) * 127.5).astype(np.uint8)).resize(size, Image.BICUBIC)
    return (np.asarray(im).astype(np.float32) / 127.5) - 1.0


def gen_textured_bg(rng, size=SIZE):
    """A mottled/marbled texture (several octaves of blurred noise
    blended together) -- the leather/marble/parchment finish real card
    frames and text boxes have, as opposed to the flat or smoothly
    gradiented backgrounds the other text generators use."""
    w, h = size
    base_color = rng.integers(30, 220, size=3).astype(np.float32)
    texture = np.zeros((h, w), dtype=np.float32)
    amplitude, total_amp = 1.0, 0.0
    for cell in (80, 40, 20, 10, 5):
        texture += amplitude * _noise_octave(rng, size, cell)
        total_amp += amplitude
        amplitude *= 0.5
    texture /= total_amp

    color_var = rng.uniform(15, 45)
    tint = rng.uniform(-10, 10, size=3)
    img = base_color[None, None, :] + texture[..., None] * (color_var + 0.3 * tint[None, None, :])
    return np.clip(img, 0, 255).astype(np.uint8)


def gen_text_on_texture(rng):
    """Both dark and light text (contrast-forced against the *local*
    background) directly on a mottled texture -- the specific hard case
    of title text embossed on a textured card frame, which text-on-flat
    or text-on-gradient don't cover: here the right color to contrast
    against changes from one letter to the next as the texture varies."""
    img = gen_textured_bg(rng)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    w, h = SIZE
    margin = int(w * 0.05)
    y = int(h * 0.03)
    while y < h - margin:
        font_size = int(rng.integers(9, 80))
        font = _random_font(rng, font_size)
        text = _random_text(rng, min_len=8, max_len=35)
        bbox = draw.textbbox((margin, y), text, font=font)
        x0, y0 = max(0, bbox[0]), max(0, bbox[1])
        x1, y1 = min(w, bbox[2]), min(h, bbox[3])
        region = img[y0:y1, x0:x1]
        bg_luminance = _luminance(region.reshape(-1, 3).mean(axis=0)) if region.size else 128.0
        color = _contrasting_color(rng, bg_luminance)
        draw.text((margin, y), text, font=font, fill=color)
        y += font_size + int(rng.integers(8, 22))
    return np.array(im)


def gen_embossed_title_on_texture(rng):
    """Title-style text: a light gray/metallic fill with a dark outline
    and drop shadow, on a warm brown/tan mottled texture -- the
    engraved-looking card title on a leather-look frame, as opposed to
    gen_text_on_texture's flat contrast-forced fill. This specific
    style (not just "text on texture" generically) has been the
    stubborn remaining failure case."""
    w, h = SIZE
    base_color = np.array([rng.uniform(90, 170), rng.uniform(55, 120), rng.uniform(25, 80)], dtype=np.float32)
    texture = np.zeros((h, w), dtype=np.float32)
    amplitude, total_amp = 1.0, 0.0
    for cell in (80, 40, 20, 10, 5):
        texture += amplitude * _noise_octave(rng, SIZE, cell)
        total_amp += amplitude
        amplitude *= 0.5
    texture /= total_amp
    color_var = rng.uniform(15, 40)
    img = np.clip(base_color[None, None, :] + texture[..., None] * color_var, 0, 255).astype(np.uint8)

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    margin = int(w * 0.06)
    y = int(h * 0.04)
    while y < h - margin:
        font_size = int(rng.integers(20, 95))
        font = _random_font(rng, font_size)
        text = _random_text(rng, min_len=6, max_len=22)
        bbox = draw.textbbox((margin, y), text, font=font)
        x0, y0 = max(0, bbox[0]), max(0, bbox[1])
        x1, y1 = min(w, bbox[2]), min(h, bbox[3])
        region = img[y0:y1, x0:x1]
        bg_luminance = _luminance(region.reshape(-1, 3).mean(axis=0)) if region.size else 128.0

        # Low-contrast fill (the "beta card" look: gray title text that's
        # only a little lighter than its brown border) legible mainly via
        # a darker outline/shadow rather than raw fill-vs-background
        # contrast -- not the wide range _contrasting_color uses elsewhere.
        fill_color = offset_color(rng, bg_luminance, (15, 55))
        stroke_color = tuple(int(c) for c in rng.integers(10, 50, size=3))
        stroke_width = int(rng.integers(1, 4))
        shadow_offset = int(rng.integers(1, 4))
        shadow_color = tuple(int(c) for c in rng.integers(0, 30, size=3))
        draw.text((margin + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color)
        draw.text((margin, y), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color)
        y += font_size + int(rng.integers(20, 50))
    return np.array(im)


def gen_text_on_gradient(rng):
    """Paragraphs of text over a smooth linear color gradient."""
    img = gen_gradient(rng)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    w, h = SIZE
    margin = int(w * 0.05)
    y = int(h * 0.03)
    while y < h - margin:
        font_size = int(rng.integers(9, 80))
        font = _random_font(rng, font_size)
        text = _random_text(rng, min_len=10, max_len=40)
        bbox = draw.textbbox((margin, y), text, font=font)
        x0, y0 = max(0, bbox[0]), max(0, bbox[1])
        x1, y1 = min(w, bbox[2]), min(h, bbox[3])
        region = img[y0:y1, x0:x1]
        bg_luminance = _luminance(region.reshape(-1, 3).mean(axis=0)) if region.size else 128.0
        color = _contrasting_color(rng, bg_luminance)
        draw.text((margin, y), text, font=font, fill=color)
        y += font_size + int(rng.integers(6, 18))
    return np.array(im)


def gen_gradient(rng):
    w, h = SIZE
    c1 = rng.integers(0, 256, size=3).astype(np.float32)
    c2 = rng.integers(0, 256, size=3).astype(np.float32)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    t = x * np.cos(angle) + y * np.sin(angle)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    img = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
    return img.astype(np.uint8)


def gen_lines(rng):
    """Straight lines of varied width and color over a neutral background."""
    w, h = SIZE
    im = Image.new("RGB", SIZE, _random_bg(rng, light=rng.random() < 0.5))
    draw = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(20, 40))):
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        width = int(rng.integers(1, 30))
        if rng.random() < 0.5:
            y = int(rng.integers(0, h))
            draw.line([(0, y), (w, y)], fill=color, width=width)
        else:
            x = int(rng.integers(0, w))
            draw.line([(x, 0), (x, h)], fill=color, width=width)
    return np.array(im)


def gen_shapes(rng):
    """A mix of circles, squares, and nested rectangular frames."""
    w, h = SIZE
    im = Image.new("RGB", SIZE, _random_bg(rng, light=rng.random() < 0.5))
    draw = ImageDraw.Draw(im)

    for _ in range(int(rng.integers(10, 20))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(15, 130)
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        filled = rng.random() < 0.4
        fill = color if filled else None
        outline_color = tuple(int(c) for c in rng.integers(0, 256, size=3)) if filled else color
        width = int(rng.integers(2, 10))
        box = [cx - r, cy - r, cx + r, cy + r]
        if rng.random() < 0.5:
            draw.ellipse(box, fill=fill, outline=outline_color, width=width)
        else:
            draw.rectangle(box, fill=fill, outline=outline_color, width=width)

    for _ in range(int(rng.integers(2, 5))):
        margin = int(min(w, h) * rng.uniform(0.03, 0.35))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        width = int(rng.integers(3, 14))
        draw.rectangle([margin, margin, w - margin, h - margin], outline=color, width=width)

    return np.array(im)


GENERATORS = [
    ("synthetic_text_light_1", lambda rng: gen_text_block(rng, light_bg=True)),
    ("synthetic_text_light_2", lambda rng: gen_text_block(rng, light_bg=True)),
    ("synthetic_text_gradient_1", gen_text_on_gradient),
    ("synthetic_text_gradient_2", gen_text_on_gradient),
    ("synthetic_text_dark_1", lambda rng: gen_text_block(rng, light_bg=False)),
    ("synthetic_text_dark_2", lambda rng: gen_text_block(rng, light_bg=False)),
    ("synthetic_text_texture_1", gen_text_on_texture),
    ("synthetic_text_texture_2", gen_text_on_texture),
    ("synthetic_text_embossed_1", gen_embossed_title_on_texture),
    ("synthetic_text_embossed_2", gen_embossed_title_on_texture),
    ("synthetic_lines_1", gen_lines),
    ("synthetic_shapes_1", gen_shapes),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic graphic-design training source images.")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "sources"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, gen) in enumerate(GENERATORS):
        rng = np.random.default_rng(args.seed + i)
        img = gen(rng)
        Image.fromarray(img).save(out_dir / f"{name}.png")
        print(f"wrote {name}.png")


if __name__ == "__main__":
    main()
