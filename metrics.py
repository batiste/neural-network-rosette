"""Image-quality metrics: whole-image PSNR/SSIM, plus a variant restricted
to high-gradient (edge/text) regions of the target.

Whole-image PSNR/SSIM are dominated by large flat/textured areas and are
known to reward blur over genuine sharpness (a network that plays it safe
scores fine; one that's sharper but very slightly misaligned can score
worse). They can't tell "great everywhere except this one bad case" apart
from "mediocre everywhere". The edge-masked variant looks only at pixels
where the target actually has strong local gradients -- text strokes,
line art, hard boundaries -- so it tracks the specific thing a blurry
model gets wrong, instead of averaging it away.
"""

import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ITU-R BT.601 luma weights: how much each channel contributes to
# perceived brightness. Human vision is far more sensitive to luminance
# (brightness/value) differences than to chrominance (hue/saturation)
# ones, but plain RGB PSNR/SSIM -- and a naive (R+G+B)/3 "grayscale" --
# treat all three channels as equally important, which they perceptually
# aren't.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


def to_luma(img):
    return (img.astype(np.float32) * LUMA_WEIGHTS).sum(axis=2)


def compute_metrics(candidate, reference):
    return {
        "psnr": float(peak_signal_noise_ratio(reference, candidate, data_range=255)),
        "ssim": float(structural_similarity(reference, candidate, channel_axis=2, data_range=255)),
    }


def compute_luma_metrics(candidate, reference):
    """PSNR/SSIM on perceived brightness alone -- the component human
    vision is most sensitive to, and which whole-image RGB metrics
    dilute by weighting it equally against far-less-noticeable color
    error."""
    c_luma, r_luma = to_luma(candidate), to_luma(reference)
    return {
        "luma_psnr": float(peak_signal_noise_ratio(r_luma, c_luma, data_range=255)),
        "luma_ssim": float(structural_similarity(r_luma, c_luma, data_range=255)),
    }


def compute_color_metrics(candidate, reference):
    """Mean perceptual color difference (Delta-E76 / CIE76) in CIE Lab
    space -- a color space designed so equal Euclidean distances
    correspond to roughly equal perceived color differences, unlike raw
    RGB error."""
    lab_c = rgb2lab(candidate.astype(np.float32) / 255.0)
    lab_r = rgb2lab(reference.astype(np.float32) / 255.0)
    delta_e = np.sqrt(((lab_c - lab_r) ** 2).sum(axis=2))
    return {"delta_e_mean": float(delta_e.mean()), "delta_e_p90": float(np.percentile(delta_e, 90))}


def edge_mask(reference, percentile=90):
    """Boolean mask of the highest-gradient pixels (by default the top
    10%) in the reference image, by perceptual luma."""
    gray = to_luma(reference)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = gx + gy
    threshold = np.percentile(grad, percentile)
    return grad >= threshold


def compute_edge_metrics(candidate, reference, percentile=90):
    mask = edge_mask(reference, percentile)
    if not mask.any():
        return {"edge_psnr": float("nan"), "edge_ssim": float("nan"), "edge_fraction": 0.0}

    candidate_f = candidate.astype(np.float64)
    reference_f = reference.astype(np.float64)
    mse = float(np.mean((candidate_f[mask] - reference_f[mask]) ** 2))
    edge_psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float("inf")

    _, ssim_map = structural_similarity(reference, candidate, channel_axis=2, data_range=255, full=True)
    edge_ssim = float(ssim_map.mean(axis=2)[mask].mean())

    return {"edge_psnr": float(edge_psnr), "edge_ssim": edge_ssim, "edge_fraction": float(mask.mean())}
