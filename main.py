import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn

from images import ImagePair

# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

SCRIPT_DIR = Path(__file__).resolve().parent


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PixelMLP(nn.Module):
    """Tiny shared MLP applied independently to each of the 9 pixels in a
    3x3 neighborhood, mapping it to the corresponding output pixel."""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def to_bytes(pixels):
    return [(255 * p[0], 255 * p[1], 255 * p[2]) for p in pixels]


def weighted_mean(p1, p2):
    return (
        (3 * p1[0] + 2 * p2[0]) / 5.0,
        (3 * p1[1] + 2 * p2[1]) / 5.0,
        (3 * p1[2] + 2 * p2[2]) / 5.0,
    )


def store_pixel(array, index, v):
    if index < 0 or index >= len(array):
        return
    if array[index] is None:
        array[index] = v
    else:
        array[index] = weighted_mean(array[index], v)


def store_pixels(array, index, pixels, width):
    pixels = to_bytes(pixels)

    store_pixel(array, index - width - 1, pixels[0])
    store_pixel(array, index - width, pixels[1])
    store_pixel(array, index - width + 1, pixels[2])

    store_pixel(array, index - 1, pixels[3])
    store_pixel(array, index, pixels[4])
    store_pixel(array, index + 1, pixels[5])

    store_pixel(array, index + width - 1, pixels[6])
    store_pixel(array, index + width, pixels[7])
    store_pixel(array, index + width + 1, pixels[8])


def interior_coords(pair):
    height, width, _ = pair.inimg.shape
    return [(x, y) for y in range(1, height - 1) for x in range(1, width - 1)]


def build_dataset(pair, coords):
    """Flattened (N*9, 3) input/target pixel arrays: the network treats
    each of the 9 neighborhood positions as an independent shared-weight
    sample, so all positions across all points are trained together."""
    inputs = np.array([pair.input_patch(x, y) for x, y in coords], dtype=np.float32)
    targets = np.array([pair.output_patch(x, y) for x, y in coords], dtype=np.float32)
    return torch.from_numpy(inputs.reshape(-1, 3)), torch.from_numpy(targets.reshape(-1, 3))


def train(pair, epochs, batch_size, lr, device, rng):
    coords = interior_coords(pair)
    inputs, targets = build_dataset(pair, coords)
    inputs, targets = inputs.to(device), targets.to(device)
    n = inputs.shape[0]

    model = PixelMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))

    print("Training neural network ...")
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator).to(device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_in, batch_target = inputs[idx], targets[idx]

            optimizer.zero_grad()
            prediction = model(batch_in)
            loss = loss_fn(prediction, batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        print(f"  epoch {epoch + 1}/{epochs}  mse={epoch_loss / n:.5f}")

    print("Neural network trained")
    return model


def write_result_image(pair, model, device, filename):
    height, width, _ = pair.inimg.shape
    oheight, owidth, _ = pair.outimg.shape
    outw = owidth - 6
    outh = oheight - 6
    assert (width - 2) * pair.scale == outw
    assert (height - 2) * pair.scale == outh

    coords = interior_coords(pair)
    batch_in = np.array([pair.input_patch(x, y) for x, y in coords], dtype=np.float32)
    batch_in = torch.from_numpy(batch_in).to(device)  # (N, 9, 3)
    with torch.no_grad():
        predictions = model(batch_in.reshape(-1, 3)).reshape(batch_in.shape[0], 9, 3)
    predictions = predictions.cpu().numpy()

    pixels = [None] * (outh * outw)
    for (x, y), prediction in zip(coords, predictions):
        index = (outw + (3 * outw * (y - 1))) + (1 + ((x - 1) * 3))
        store_pixels(pixels, index, prediction, outw)

    a = np.array(pixels, dtype=np.uint8).reshape(outh, outw, 3)
    imageio.imwrite(filename, a)


def write_check_image(pair, model, device, rng, filename):
    """Diagnostic image: for two random sample points, show the input
    neighborhood, the clean target block, and the network's current
    prediction for that neighborhood -- 3 rows of 9 pixels per point."""
    height, width, _ = pair.inimg.shape
    rows = []
    for _ in range(2):
        x = int(rng.integers(1, width - 2))
        y = int(rng.integers(1, height - 2))
        input_patch = pair.input_patch(x, y)
        output_patch = pair.output_patch(x, y)
        with torch.no_grad():
            batch_in = torch.tensor(input_patch, dtype=torch.float32, device=device)
            prediction = model(batch_in).cpu().numpy()

        rows.append(to_bytes(input_patch))
        rows.append(to_bytes(output_patch))
        rows.append(to_bytes(prediction))

    pixels = [p for row in rows for p in row]
    a = np.array(pixels, dtype=np.uint8).reshape(6, 9, 3)
    imageio.imwrite(filename, a)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny per-pixel neural network to upscale an image."
    )
    parser.add_argument(
        "--input", default=str(SCRIPT_DIR / "inkami.png"),
        help="low quality training input image (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default=str(SCRIPT_DIR / "outkami.png"),
        help="clean training target image, scale times the input size (default: %(default)s)",
    )
    parser.add_argument(
        "--scale", type=int, default=3,
        help="upscale factor; only 3 is currently supported (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for reproducible training (default: unseeded)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="number of passes over the training data (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="training minibatch size (default: %(default)s)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2,
        help="Adam learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device to train on: cpu/mps/cuda (default: auto-detect, prefers MPS on Apple Silicon)",
    )
    parser.add_argument(
        "--result", default="result.png",
        help="where to write the upscaled result image (default: %(default)s)",
    )
    parser.add_argument(
        "--check", default="check.png",
        help="where to write the diagnostic check image (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    pair = ImagePair.load(args.input, args.output, scale=args.scale)
    model = train(pair, args.epochs, args.batch_size, args.lr, device, rng)

    write_check_image(pair, model, device, rng, args.check)
    print(f"Check image written to {args.check}")

    print(f"Outputting the result image to {args.result}")
    write_result_image(pair, model, device, args.result)


if __name__ == "__main__":
    main()
