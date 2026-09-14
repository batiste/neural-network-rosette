"""Synthetic degradation pipeline: turns a clean high-resolution image patch
into a paired low-quality "scan" with offset-printing-style moire artifacts.

The core physical idea: real moire comes from aliasing between a printed
halftone screen and a scanner/sensor grid. We simulate that by painting a
per-channel (CMY) halftone dot screen onto the clean high-res patch, adding
a lower-frequency sinusoidal interference ripple, shifting channels to mimic
press misregistration, and then downsampling with a method that does not
fully anti-alias -- so the fine screen pattern survives as visible moire
bands instead of being smoothed away. Scan/camera-like post-processing
(blur, sharpening halos, noise, JPEG) is applied afterwards.
"""

import io

import numpy as np
from PIL import Image, ImageFilter


PRESETS = {
    "subtle_scan": dict(
        halftone_strength=(0.05, 0.15),
        halftone_period=(4.0, 7.0),
        ripple_strength=(0.05, 0.15),
        misregistration_px=(0.0, 0.5),
        downsample_method=("box",),
        scan_dpi_factor=(0.8, 1.3),
        blur_radius=(0.2, 0.6),
        sharpen_percent=(0, 60),
        noise_sigma=(1.0, 4.0),
        jpeg_quality=(75, 95),
    ),
    "severe_offset_print": dict(
        halftone_strength=(0.35, 0.6),
        halftone_period=(2.5, 4.5),
        ripple_strength=(0.1, 0.25),
        misregistration_px=(0.5, 2.0),
        downsample_method=("nearest", "box"),
        scan_dpi_factor=(0.5, 1.0),
        blur_radius=(0.0, 0.4),
        sharpen_percent=(50, 180),
        noise_sigma=(2.0, 8.0),
        jpeg_quality=(50, 85),
    ),
    "colored_channel": dict(
        halftone_strength=(0.2, 0.4),
        halftone_period=(3.0, 6.0),
        ripple_strength=(0.05, 0.2),
        misregistration_px=(1.0, 3.0),
        downsample_method=("nearest", "box"),
        scan_dpi_factor=(0.6, 1.4),
        blur_radius=(0.0, 0.3),
        sharpen_percent=(0, 100),
        noise_sigma=(1.0, 5.0),
        jpeg_quality=(60, 90),
    ),
    "mixed": dict(
        halftone_strength=(0.05, 0.6),
        halftone_period=(2.5, 7.0),
        ripple_strength=(0.0, 0.25),
        misregistration_px=(0.0, 3.0),
        downsample_method=("nearest", "box", "bilinear"),
        scan_dpi_factor=(0.4, 2.0),
        blur_radius=(0.0, 0.6),
        sharpen_percent=(0, 180),
        noise_sigma=(0.0, 8.0),
        jpeg_quality=(45, 95),
    ),
}

DOWNSAMPLE_FILTERS = {
    "nearest": Image.NEAREST,
    "box": Image.BOX,
    "bilinear": Image.BILINEAR,
}


def _sample(rng, bounds):
    if isinstance(bounds, tuple) and len(bounds) == 2 and all(isinstance(b, (int, float)) for b in bounds):
        return rng.uniform(bounds[0], bounds[1])
    return rng.choice(bounds)


def sinusoidal_ripple(h, w, rng, strength):
    """m(x, y) = sin(2*pi*f1*(x*cos(t1)+y*sin(t1))+p1) * sin(2*pi*f2*(...)+p2)
    Two overlaid gratings at close frequencies/angles produce a slowly
    varying interference beat pattern -- the classic moire look."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    def wave():
        f = rng.uniform(0.15, 0.45)
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        return np.sin(2 * np.pi * f * (x * np.cos(theta) + y * np.sin(theta)) + phi)

    m = wave() * wave()
    return 1.0 + strength * m


def halftone_screen_distance(h, w, period, angle_deg, rng):
    """Per-pixel normalized distance to the nearest halftone dot center on a
    rotated grid of the given period/angle."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = np.deg2rad(angle_deg)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    phase_x = rng.uniform(0, period)
    phase_y = rng.uniform(0, period)
    cx = np.mod(xr + phase_x, period) - period / 2
    cy = np.mod(yr + phase_y, period) - period / 2
    return np.sqrt(cx ** 2 + cy ** 2) / (period / 2)


def apply_halftone(rgb01, period, strength, rng):
    """Overlay an AM halftone dot screen per CMY-ish channel, each at its
    own rotation angle (the offset-print rosette), sized by local ink
    coverage. rgb01 is a float array in [0, 1]."""
    h, w = rgb01.shape[:2]
    cmy = 1.0 - rgb01
    angles = [15.0, 75.0, 0.0]
    out_cmy = np.empty_like(cmy)
    for c in range(3):
        angle = angles[c] + rng.uniform(-5, 5)
        dist = halftone_screen_distance(h, w, period, angle, rng)
        coverage = np.clip(cmy[..., c], 0, 1)
        dot_radius = np.sqrt(coverage)
        edge = 0.15
        ink_mask = np.clip((dot_radius - dist) / edge + 0.5, 0, 1)
        halftoned = ink_mask
        out_cmy[..., c] = coverage * (1 - strength) + halftoned * strength
    return np.clip(1.0 - out_cmy, 0, 1)


def apply_misregistration(rgb_img, max_px, rng):
    """Shift each channel by an independent small sub-pixel offset, as in a
    press where the C/M/Y/K plates are not perfectly aligned."""
    if max_px <= 0:
        return rgb_img
    channels = []
    for c in range(3):
        dx = rng.uniform(-max_px, max_px)
        dy = rng.uniform(-max_px, max_px)
        im = Image.fromarray(rgb_img[..., c])
        shifted = im.transform(im.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.BICUBIC)
        channels.append(np.array(shifted))
    return np.stack(channels, axis=-1)


def degrade(clean_hr, scale, rng, preset="mixed"):
    """clean_hr: uint8 (H, W, 3) array, H and W divisible by `scale`.
    Returns a uint8 (H/scale, W/scale, 3) degraded low-res array."""
    params = PRESETS[preset]
    h, w = clean_hr.shape[:2]

    rgb01 = clean_hr.astype(np.float32) / 255.0

    halftone_strength = _sample(rng, params["halftone_strength"])
    if halftone_strength > 0:
        period = _sample(rng, params["halftone_period"])
        rgb01 = apply_halftone(rgb01, period, halftone_strength, rng)

    ripple_strength = _sample(rng, params["ripple_strength"])
    if ripple_strength > 0:
        rgb01 = np.clip(rgb01 * sinusoidal_ripple(h, w, rng, ripple_strength)[..., None], 0, 1)

    hr_uint8 = (rgb01 * 255.0).round().astype(np.uint8)
    hr_uint8 = apply_misregistration(hr_uint8, _sample(rng, params["misregistration_px"]), rng)

    lr_w, lr_h = w // scale, h // scale

    # Simulate scanning/photographing the print at a different effective
    # resolution than the final output: a coarser sensor grid samples the
    # halftone screen sparsely, aliasing into broad low-frequency bands; a
    # finer one resolves individual dots, giving a subtler, higher-frequency
    # texture. Either way we resample back to the fixed output size after.
    dpi_factor = _sample(rng, params.get("scan_dpi_factor", (1.0, 1.0)))
    scan_w = max(1, round(lr_w * dpi_factor))
    scan_h = max(1, round(lr_h * dpi_factor))

    method = DOWNSAMPLE_FILTERS[_sample(rng, params["downsample_method"])]
    scan_img = Image.fromarray(hr_uint8).resize((scan_w, scan_h), resample=method)
    lr_img = scan_img if (scan_w, scan_h) == (lr_w, lr_h) else scan_img.resize((lr_w, lr_h), resample=Image.BICUBIC)

    blur_radius = _sample(rng, params["blur_radius"])
    if blur_radius > 0:
        lr_img = lr_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    sharpen_percent = _sample(rng, params["sharpen_percent"])
    if sharpen_percent > 0:
        lr_img = lr_img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(sharpen_percent), threshold=2))

    lr = np.array(lr_img).astype(np.float32)

    noise_sigma = _sample(rng, params["noise_sigma"])
    if noise_sigma > 0:
        lr = lr + rng.normal(0, noise_sigma, size=lr.shape)
    lr = np.clip(lr, 0, 255).astype(np.uint8)

    jpeg_quality = int(_sample(rng, params["jpeg_quality"]))
    buf = io.BytesIO()
    Image.fromarray(lr).save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    lr = np.array(Image.open(buf).convert("RGB"))

    return lr
