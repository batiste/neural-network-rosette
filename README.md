# neural-network-rosette

Try to use a neural network to enlarge a low quality scan 3X times.

My original idea was to "fix" scans of MTG cards that contains the typical moiré pattern typical of offest printing.

The problem was to find good training data (clean image + moiré image) that would align perfectly has training data. The fix: synthesize the moiré instead of scanning for it — degrade clean, high-resolution digital artwork with a simulated offset-print pattern, and train on that paired data.

## How it works

- Clean, high-resolution digital artwork (`sources/`) is used as training targets — no real scans needed.
- Random crops of those images are synthetically degraded (`moire.py`) to simulate offset-print moiré: a per-channel halftone screen at rosette angles, sinusoidal interference banding, channel misregistration, variable effective scan resolution, blur, sharpening halos, sensor noise, and JPEG artifacts.
- A small residual CNN (`model.py`) is trained to predict a correction on top of a bicubic upscale, giving it enough spatial context (a ~31x31 pixel receptive field, vs. a naive per-pixel approach's 3x3) to actually recognize and remove the pattern instead of guessing per-pixel.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Synthetic moiré presets

`generate_moire.py` samples random crops from `sources/` and runs each through every degradation preset, so the parameters can be inspected before training:

```
.venv/bin/python generate_moire.py --seed 1 --out moire_preview.png
```

![Moire presets](moire_preview.png)

## Results

`results.py` runs the trained model on fresh synthetic crops: source / degraded moiré input / network output.

```
.venv/bin/python results.py --checkpoint checkpoints/best.pt --out results.png
```

![Results](results.png)

## Training and inference

```
.venv/bin/python train.py --epochs 30 --checkpoint-dir checkpoints    # train (auto-uses MPS on Apple Silicon)
.venv/bin/python infer.py --checkpoint checkpoints/best.pt --input some_image.png --output upscaled.png
```

`train.py` holds out the last `--val-count` images in `sources/` for validation, logs PSNR/SSIM per epoch, and writes a preview panel (input/bicubic/prediction/target) every `--preview-every` epochs plus `latest.pt`/`best.pt` checkpoints under `--checkpoint-dir`. `checkpoints/best.pt` is committed to the repo (~5.3MB) so `infer.py`/`results.py` work out of the box without retraining; everything else under `checkpoints/` (per-epoch previews, optimizer state, training history) is gitignored. The committed `best.pt` reached val PSNR 27.9 / SSIM 0.79 after 30 epochs on 9 source images.

`sources/` holds the clean training images. The two small `.jpg` files are tracked in git; the larger `.png` ones are gitignored (`*.png`) since they're multi-megabyte originals — keep your own copies there locally, `train.py` just needs at least a few images in that directory.

Run any script with `--help` for the full set of options.
