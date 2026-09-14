"""On-the-fly training data: random crops from the clean source images,
degraded synthetically with moire.degrade(). Split by file so validation
uses images the model never trains on."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from graphics import add_synthetic_graphics
from moire import PRESETS, degrade

EXTS = {".png", ".jpg", ".jpeg"}


def list_sources(source_dir):
    return sorted(p for p in Path(source_dir).iterdir() if p.suffix.lower() in EXTS)


def split_sources(sources, val_count=2):
    """Hold out val_count images for validation, stratified so both the
    photographic/painterly sources and the generate_synthetic_sources.py
    images (prefixed "synthetic_") are represented -- a plain tail slice
    over the sorted list would put zero synthetic images in validation,
    since they all sort near the end alphabetically."""
    if val_count < 1:
        raise ValueError(f"val_count must be at least 1 (got {val_count})")
    if len(sources) <= val_count:
        raise ValueError(
            f"Need more than val_count={val_count} source images to hold any out for training "
            f"(found {len(sources)})."
        )

    synthetic = [p for p in sources if p.name.startswith("synthetic_")]
    real = [p for p in sources if not p.name.startswith("synthetic_")]

    n_synthetic = min(len(synthetic), max(1, val_count // 2)) if synthetic else 0
    n_real = min(len(real), val_count - n_synthetic)
    n_synthetic = min(len(synthetic), val_count - n_real)

    val_paths = real[-n_real:] + synthetic[-n_synthetic:]
    val_names = {p.name for p in val_paths}
    train_paths = [p for p in sources if p.name not in val_names]
    return train_paths, val_paths


def _area_weights(images, paths, synthetic_share=0.3):
    """Per-image sampling probability: weighted by pixel area *within* the
    real/synthetic groups (so a 30 Mpx painting doesn't get the same
    selection probability as a 1.4 Mpx synthetic page -- equal-per-file
    sampling would oversample the smaller group by >20x relative to its
    actual content), but with an explicit, deliberate probability mass
    reserved for the synthetic group overall (since we *do* want to
    oversample graphic-design content relative to its true pixel share
    -- it's just a dial we should set on purpose, not a side effect of
    how many files happen to be in each group)."""
    is_synthetic = np.array([p.name.startswith("synthetic_") for p in paths])
    areas = np.array([img.shape[0] * img.shape[1] for img in images], dtype=np.float64)

    weights = np.zeros_like(areas)
    for is_synth, share in ((True, synthetic_share), (False, 1 - synthetic_share)):
        group = is_synthetic == is_synth
        if group.any():
            weights[group] = share * areas[group] / areas[group].sum()
    return weights / weights.sum()


class MoireDataset(Dataset):
    """Each epoch is a virtual `epoch_length` random samples drawn from the
    given source images. `deterministic=True` seeds each index so the same
    crop/degradation is reproduced every time -- used for validation."""

    def __init__(self, paths, hr_size, scale, epoch_length, presets=None, deterministic=False, seed=0,
                 graphics_prob=0.5, synthetic_share=0.3):
        self.paths = list(paths)
        self.images = [np.array(Image.open(p).convert("RGB")) for p in self.paths]
        for img, path in zip(self.images, self.paths):
            if img.shape[0] < hr_size or img.shape[1] < hr_size:
                raise ValueError(f"{path} is smaller than hr_size={hr_size}")
        self.hr_size = hr_size
        self.scale = scale
        self.epoch_length = epoch_length
        self.presets = presets or list(PRESETS)
        self.deterministic = deterministic
        self.seed = seed
        self.graphics_prob = graphics_prob
        self.weights = _area_weights(self.images, self.paths, synthetic_share=synthetic_share)

    def __len__(self):
        return self.epoch_length

    def _rng(self, idx):
        if self.deterministic:
            return np.random.default_rng(self.seed + idx)
        return np.random.default_rng()

    def __getitem__(self, idx):
        rng = self._rng(idx)
        img = self.images[int(rng.choice(len(self.images), p=self.weights))]
        h, w, _ = img.shape
        hr_size = self.hr_size
        x = int(rng.integers(0, w - hr_size + 1))
        y = int(rng.integers(0, h - hr_size + 1))
        hr = img[y:y + hr_size, x:x + hr_size]

        if rng.random() < 0.5:
            hr = hr[:, ::-1]
        if rng.random() < 0.5:
            hr = hr[::-1, :]
        hr = np.ascontiguousarray(hr)

        protect_mask = None
        if rng.random() < self.graphics_prob:
            hr, protect_mask = add_synthetic_graphics(hr, rng)

        preset = self.presets[int(rng.integers(0, len(self.presets)))]
        lr = degrade(hr, self.scale, rng, preset=preset, protect_mask=protect_mask)

        hr_t = torch.from_numpy(hr.astype(np.float32).transpose(2, 0, 1) / 255.0)
        lr_t = torch.from_numpy(lr.astype(np.float32).transpose(2, 0, 1) / 255.0)
        return lr_t, hr_t
