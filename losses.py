"""Loss functions for training. Plain pixel L1 structurally favors the
"safe average" on fine detail the model is uncertain about -- the same
regression-to-the-mean blur bias that motivates adversarial/perceptual
losses in the super-resolution literature. A Laplacian pyramid loss is a
simpler, self-contained way to push back against it (no pretrained
network needed): it penalizes prediction-vs-target mismatch separately
at each frequency band, including the finest ones where blur shows up
first, instead of averaging all frequencies into one pixel-space error.
"""

import torch.nn.functional as F


def gaussian_pyramid(img, levels):
    pyramid = [img]
    current = img
    for _ in range(levels - 1):
        current = F.avg_pool2d(current, 2)
        pyramid.append(current)
    return pyramid


def laplacian_pyramid(img, levels):
    gauss = gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels - 1):
        upsampled = F.interpolate(gauss[i + 1], size=gauss[i].shape[-2:], mode="bilinear", align_corners=False)
        lap.append(gauss[i] - upsampled)
    lap.append(gauss[-1])  # residual low-frequency band
    return lap


def laplacian_pyramid_loss(pred, target, levels=4):
    lap_pred = laplacian_pyramid(pred, levels)
    lap_target = laplacian_pyramid(target, levels)
    return sum(F.l1_loss(lp, lt) for lp, lt in zip(lap_pred, lap_target)) / levels
