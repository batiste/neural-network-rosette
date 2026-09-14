# neural-network-rosette

Try to use a neural network to enlarge a low quality scan 3X times.

My original idea was to "fix" scans of MTG cards that contains the typical moiré pattern typical of offest printing.

The problem was to find good training data (clean image + moiré image) that would align perfectly has training data. The fix: synthesize the moiré instead of scanning for it — degrade clean, high-resolution digital artwork with a simulated offset-print pattern, and train on that paired data.

## How it works

- Clean, high-resolution digital artwork (`sources/`) is used as training targets — no real scans needed.
- Random crops of those images are synthetically degraded (`moire.py`) to simulate offset-print moiré: a per-channel halftone screen at rosette angles, sinusoidal interference banding, channel misregistration, variable effective scan resolution, blur, sharpening halos, sensor noise, and JPEG artifacts.
- Some crops get random text/line overlays (`graphics.py`) before degrading, and `sources/` also includes procedurally generated text/gradient/shape images (`generate_synthetic_sources.py`) — the photographic art sources have essentially no sharp graphic-design content (typography, hard-ruled lines) on their own, so without this the model never learns to reconstruct it.
- A small residual CNN (`model.py`) is trained to predict a correction on top of a bicubic upscale, giving it enough spatial context (a ~31x31 pixel receptive field, vs. a naive per-pixel approach's 3x3) to actually recognize and remove the pattern instead of guessing per-pixel.
- Besides whole-image PSNR/SSIM, `metrics.py` tracks an edge-masked variant (computed only on the highest-gradient ~10% of pixels — text strokes, line art, hard boundaries) since whole-image averages are dominated by large flat/textured areas and are known to reward blur over genuine sharpness. `train.py` selects `best.pt` by edge PSNR, not whole-image PSNR.

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

## Real-world results

The above is all synthetic — the actual test is real photos of physical cards, not synthetically degraded crops. `real_world_results.py` runs inference on a couple of real card photos and panels source / Lanczos 3x (the strongest classical resampling filter, per `evaluate.py`) / network output side by side:

```
.venv/bin/python infer.py --checkpoint checkpoints/best.pt --input bear.webp --output bear_upscaled.png
.venv/bin/python infer.py --checkpoint checkpoints/best.pt --input lotus.webp --output lotus_upscaled.png
.venv/bin/python real_world_results.py --out real_world_results.png
```

![Real-world results](real_world_results.png)

The image above is pasted at full native resolution (4800x5760, lossless) but GitHub's inline preview downsamples it to fit the viewport — **[download the full-resolution PNG](https://raw.githubusercontent.com/batiste/neural-network-rosette/master/real_world_results.png)** (~29MB) to see it at actual pixel size, or open it locally after cloning. Note that Lanczos, having done no denoising, just enlarges the source's existing noise/grain along with it — the network output is visibly cleaner at comparable letterform sharpness, which is really the network's main real contribution here rather than out-sharpening a good classical filter on edges alone.

Illustration detail and body/rules text come out clearly sharper than the source at this scale. The one honest weak point: small embossed title text (light gray on a textured card border) hasn't improved as much as everything else across several rounds of tuning — it's plausibly close to an information floor for what a model this size, trained this way, can confidently reconstruct from the signal actually present in the source photo, rather than something more data or training would keep fixing.

## Training and inference

```
.venv/bin/python train.py --epochs 30 --checkpoint-dir checkpoints    # train (auto-uses MPS on Apple Silicon)
.venv/bin/python infer.py --checkpoint checkpoints/best.pt --input some_image.png --output upscaled.png
```

`train.py` holds out `--val-count` images in `sources/` for validation (stratified between photographic and synthetic sources, see `dataset.split_sources`), logs PSNR/SSIM and edge PSNR/SSIM per epoch, and writes a preview panel (input/bicubic/prediction/target) every `--preview-every` epochs plus `latest.pt`/`best.pt` checkpoints under `--checkpoint-dir`. `checkpoints/best.pt` is committed to the repo (~5.3MB) so `infer.py`/`results.py` work out of the box without retraining; everything else under `checkpoints/` (per-epoch previews, optimizer state, training history) is gitignored.

`sources/` holds the clean training images: photographic/painterly artwork plus the procedurally generated text/gradient/line/shape images from `generate_synthetic_sources.py` (regenerate anytime with a different `--seed`; nothing there is hand-crafted, so it isn't committed either). The small `.jpg` files are tracked in git; the larger `.png` ones are gitignored (`*.png`) since they're multi-megabyte originals — keep your own copies there locally, `train.py` just needs at least a handful of images in that directory.

Run any script with `--help` for the full set of options.
