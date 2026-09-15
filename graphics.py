"""Synthetic graphic-design overlays (text and straight lines/borders)
composited onto clean training crops before degradation.

The source images are all painterly digital illustration -- the network
never otherwise sees a letterform or a hard ruled edge, so it has no
learned prior for reconstructing them and just smooths them like brush
texture. Overlaying random text and lines onto some training crops gives
it that exposure directly, without needing new source assets.
"""

import logging
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

logging.getLogger("fontTools").setLevel(logging.ERROR)

FONT_DIRS = [
    Path("/System/Library/Fonts/Supplemental"),
    Path("/System/Library/Fonts"),
    Path("/usr/share/fonts"),
]

_FONT_PATHS = None

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,'\":-"

# Symbol/dingbat/decorative fonts don't have normal Latin glyphs and just
# render as tofu boxes -- exclude the common ones by name as a first,
# cheap pass (the render-based check below catches the rest).
_FONT_DENYLIST = ("wingding", "webding", "symbol", "ornament", "emoji", "bookshelf")


def _font_renders_latin(path):
    """Check the font's cmap table directly for every character
    _random_text can draw -- the authoritative source of what a font
    actually supports, unlike inferring it from rendered pixels/bboxes
    (which can false-positive on coincidental size matches, or
    false-negative on fonts with unusual but real glyph shapes)."""
    try:
        cmap = TTFont(str(path), fontNumber=0, lazy=True).getBestCmap()
        if cmap is None:
            return False
        return all(ord(ch) in cmap for ch in _ALPHABET if not ch.isspace())
    except Exception:
        return False


def _discover_fonts():
    global _FONT_PATHS
    if _FONT_PATHS is not None:
        return _FONT_PATHS
    paths = []
    for d in FONT_DIRS:
        if d.is_dir():
            paths.extend(sorted(d.glob("*.ttf")))
    paths = [p for p in paths if not any(bad in p.name.lower() for bad in _FONT_DENYLIST)]
    paths = [p for p in paths if _font_renders_latin(p)]
    _FONT_PATHS = paths
    return _FONT_PATHS


def _random_font(rng, size):
    fonts = _discover_fonts()
    if fonts:
        path = fonts[int(rng.integers(0, len(fonts)))]
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            pass
    return ImageFont.load_default()


# Beta-era MTG card titles were set in Plantin, a commercial Monotype
# serif that isn't available as a system font here (and isn't something
# to source from the web). Times New Roman is the closest old-style book
# serif actually on this machine; Georgia/Big Caslon add a little
# variety while staying in the same family of look.
_TITLE_FONT_NAMES = (
    "Times New Roman.ttf", "Times New Roman Bold.ttf", "Times New Roman Italic.ttf",
    "Georgia.ttf", "Georgia Bold.ttf", "BigCaslon.ttf",
)
_TITLE_FONT_PATHS = None


def _discover_title_fonts():
    global _TITLE_FONT_PATHS
    if _TITLE_FONT_PATHS is not None:
        return _TITLE_FONT_PATHS
    by_name = {p.name: p for p in _discover_fonts()}
    paths = [by_name[n] for n in _TITLE_FONT_NAMES if n in by_name]
    _TITLE_FONT_PATHS = paths or _discover_fonts()
    return _TITLE_FONT_PATHS


def random_title_font(rng, size):
    """A font from the curated title-appropriate pool (see
    _TITLE_FONT_NAMES), for card-title-style text specifically -- as
    opposed to _random_font's full assorted pool used for generic body
    text/overlays."""
    fonts = _discover_title_fonts()
    if fonts:
        path = fonts[int(rng.integers(0, len(fonts)))]
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            pass
    return ImageFont.load_default()


def _random_text(rng, min_len=3, max_len=18):
    length = int(rng.integers(min_len, max_len + 1))
    return "".join(_ALPHABET[i] for i in rng.integers(0, len(_ALPHABET), size=length))


def _luminance(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def offset_color(rng, bg_luminance, delta_range):
    """A color offset from bg_luminance by a random amount (within
    delta_range) and direction (whichever side has room, or either if
    both do)."""
    delta = rng.uniform(*delta_range)
    room_above = 255 - bg_luminance
    room_below = bg_luminance
    if room_above >= delta and (room_below < delta or rng.random() < 0.5):
        target = bg_luminance + delta
    else:
        target = bg_luminance - delta
    target = float(np.clip(target, 0, 255))
    channel_var = rng.uniform(-15, 15, size=3)
    return tuple(int(c) for c in np.clip(target + channel_var, 0, 255))


def _contrasting_color(rng, bg_luminance):
    """A color offset from bg_luminance across a wide contrast range.

    Real card typography spans a wide contrast range -- stark black
    rules text on a cream box, but also subtle embossed/metallic titles
    that sit much closer in tone to their background. Always forcing
    near-black-or-near-white text (the previous behavior) taught the
    model an unrealistically narrow, maximal-contrast-only distribution.
    A minimum delta still keeps every sample legible."""
    return offset_color(rng, bg_luminance, (35, 170))


def add_text_overlay(img, rng, max_snippets=3):
    """Draw a few random text snippets at random position/size, with a
    color forced to contrast against whatever is actually behind it (the
    plate if one is drawn, otherwise the real local image content) --
    two independently random colors are often low-contrast by chance,
    which would train the model to treat low-contrast text as noise.

    Returns (image, mask) where mask is True over the text (+ plate, if
    any) footprint -- real printed text is solid ink, not halftoned, so
    moire.degrade() should skip halftoning it too."""
    im = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    for _ in range(int(rng.integers(1, max_snippets + 1))):
        size = int(rng.integers(max(8, h // 20), max(9, h // 6)))
        font = _random_font(rng, size)
        text = _random_text(rng)
        x = int(rng.integers(0, max(1, w - size)))
        y = int(rng.integers(0, max(1, h - size)))
        bbox = draw.textbbox((x, y), text, font=font)

        if rng.random() < 0.5:
            plate_luminance = rng.uniform(0, 255)
            plate_color = tuple(
                int(c) for c in np.clip(plate_luminance + rng.uniform(-15, 15, size=3), 0, 255)
            )
            pad = max(2, size // 6)
            alpha = int(rng.integers(160, 240))
            px0, py0 = max(0, bbox[0] - pad), max(0, bbox[1] - pad)
            px1, py1 = min(w, bbox[2] + pad), min(h, bbox[3] + pad)
            draw.rectangle([px0, py0, px1, py1], fill=plate_color + (alpha,))
            mask[py0:py1, px0:px1] = True
            bg_luminance = plate_luminance
        else:
            x0, y0 = max(0, bbox[0]), max(0, bbox[1])
            x1, y1 = min(w, bbox[2]), min(h, bbox[3])
            region = img[y0:y1, x0:x1]
            bg_luminance = _luminance(region.reshape(-1, 3).mean(axis=0)) if region.size else 128.0
            mask[y0:y1, x0:x1] = True

        color = _contrasting_color(rng, bg_luminance)
        draw.text((x, y), text, font=font, fill=color)

    return np.array(im), mask


def add_line_overlay(img, rng, max_lines=4):
    """Draw a few random straight horizontal/vertical lines and thin
    rectangle borders -- the hard ruled edges of a card frame/text box.
    Returns (image, mask) where mask is True over the drawn line pixels
    (also solid ink in real printing, so not halftoned either)."""
    im = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(im)
    h, w = img.shape[:2]
    mask_im = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask_im)

    for _ in range(int(rng.integers(1, max_lines + 1))):
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        thickness = int(rng.integers(1, max(2, min(h, w) // 60)))
        if rng.random() < 0.5:
            y = int(rng.integers(0, h))
            draw.line([(0, y), (w, y)], fill=color, width=thickness)
            mask_draw.line([(0, y), (w, y)], fill=255, width=thickness)
        else:
            x = int(rng.integers(0, w))
            draw.line([(x, 0), (x, h)], fill=color, width=thickness)
            mask_draw.line([(x, 0), (x, h)], fill=255, width=thickness)

    if rng.random() < 0.4:
        margin = int(min(h, w) * rng.uniform(0.05, 0.2))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        thickness = int(rng.integers(1, max(2, min(h, w) // 80)))
        draw.rectangle([margin, margin, w - margin, h - margin], outline=color, width=thickness)
        mask_draw.rectangle([margin, margin, w - margin, h - margin], outline=255, width=thickness)

    return np.array(im), np.array(mask_im) > 0


def add_synthetic_graphics(img, rng, text_prob=0.5, line_prob=0.5):
    """img: uint8 (H, W, 3) array. Returns (possibly-modified copy, mask)
    -- mask is True where text/line pixels were drawn."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if rng.random() < line_prob:
        img, line_mask = add_line_overlay(img, rng)
        mask |= line_mask
    if rng.random() < text_prob:
        img, text_mask = add_text_overlay(img, rng)
        mask |= text_mask
    return img, mask
