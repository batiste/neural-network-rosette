# neural-network-rosette

Try to use a neural network to enlarge a low quality scan 3X times.

My original idea was to "fix" scans of MTG cards that contains the typical moiré pattern typical of offest printing.

The problem was to find good training data (clean image + moiré image) that would align perfectly has training data. So at the end I just did use simple scaled down image and forgot about the moiré.

[Moiré effect](moire_preview.png)

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## CNN pipeline (current approach)

The single-image-pair, 3x3-neighborhood approach below turned out to be a dead end: it never beat plain bicubic/Lanczos resizing on PSNR/SSIM (see `evaluate.py`), because a 3x3 receptive field simply can't recognize a spatial interference pattern like moiré. The current approach instead:

- Uses clean, high-resolution digital artwork (`sources/`) as training targets — no real scans needed.
- Synthetically degrades random crops of those images (`moire.py`) to simulate offset-print moiré: a per-channel halftone screen at rosette angles, sinusoidal interference banding, channel misregistration, variable effective scan resolution, blur, sharpening halos, sensor noise, and JPEG artifacts.
- Trains a small residual CNN (`model.py`) that predicts a correction on top of a bicubic upscale, giving it enough spatial context to actually recognize and remove the pattern instead of guessing per-pixel.

```
.venv/bin/python generate_moire.py --seed 1 --out moire_preview.png   # inspect the synthetic degradations
.venv/bin/python train.py --epochs 30 --checkpoint-dir checkpoints    # train (auto-uses MPS on Apple Silicon)
.venv/bin/python infer.py --checkpoint checkpoints/best.pt --input some_image.png --output upscaled.png
```

`train.py` holds out the last `--val-count` images in `sources/` for validation, logs PSNR/SSIM per epoch, and writes a preview panel (input/bicubic/prediction/target) every `--preview-every` epochs plus `latest.pt`/`best.pt` checkpoints, all under `--checkpoint-dir` (gitignored — they're large binaries, not source). Run any script with `--help` for the full set of options.

## Original per-pixel approach (superseded, kept as a baseline)

The first version trained a tiny MLP on a single 100x100 input/output image pair, treating each of the 9 pixels in a 3x3 neighborhood as an independent training sample mapped to one output pixel. It's a nice minimal starting point but architecturally can't see anything larger than 3x3, so it can't learn to recognize or remove a spatial pattern like moiré.

```
.venv/bin/python main.py --seed 42 --result result.png
.venv/bin/python evaluate.py --neural result.png --panel comparison.png
```

`main.py` trains on `inkami.png`/`outkami.png` and writes `result.png` + a diagnostic `check.png`. `evaluate.py` compares any such result against classical resize baselines (nearest/bilinear/bicubic/Lanczos/sharpened-bicubic) with PSNR/SSIM and a labeled side-by-side panel. Only a 3x scale is supported, since the network's patch sizes are tied to it.


