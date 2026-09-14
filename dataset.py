"""On-the-fly training data: random crops from the clean source images,
degraded synthetically with moire.degrade(). Split by file so validation
uses images the model never trains on."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from moire import PRESETS, degrade

EXTS = {".png", ".jpg", ".jpeg"}


def list_sources(source_dir):
    return sorted(p for p in Path(source_dir).iterdir() if p.suffix.lower() in EXTS)


def split_sources(sources, val_count=2):
    if val_count < 1:
        raise ValueError(f"val_count must be at least 1 (got {val_count})")
    if len(sources) <= val_count:
        raise ValueError(
            f"Need more than val_count={val_count} source images to hold any out for training "
            f"(found {len(sources)})."
        )
    split = len(sources) - val_count
    return sources[:split], sources[split:]


class MoireDataset(Dataset):
    """Each epoch is a virtual `epoch_length` random samples drawn from the
    given source images. `deterministic=True` seeds each index so the same
    crop/degradation is reproduced every time -- used for validation."""

    def __init__(self, paths, hr_size, scale, epoch_length, presets=None, deterministic=False, seed=0):
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

    def __len__(self):
        return self.epoch_length

    def _rng(self, idx):
        if self.deterministic:
            return np.random.default_rng(self.seed + idx)
        return np.random.default_rng()

    def __getitem__(self, idx):
        rng = self._rng(idx)
        img = self.images[int(rng.integers(0, len(self.images)))]
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

        preset = self.presets[int(rng.integers(0, len(self.presets)))]
        lr = degrade(hr, self.scale, rng, preset=preset)

        hr_t = torch.from_numpy(hr.astype(np.float32).transpose(2, 0, 1) / 255.0)
        lr_t = torch.from_numpy(lr.astype(np.float32).transpose(2, 0, 1) / 255.0)
        return lr_t, hr_t
