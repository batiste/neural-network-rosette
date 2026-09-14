"""Load an input/output training image pair and extract the normalized
3x3 pixel neighborhoods used to train and query the neural network."""

import imageio.v2 as imageio

CORNER_WEIGHT = 0.42
EDGE_WEIGHT = 0.36


def pixel_to_3(p):
    return (p[0], p[1], p[2])


def normalize(p):
    return (p[0] / 255.0, p[1] / 255.0, p[2] / 255.0)


def _blend(neighbor, center, weight):
    inv = 1 - weight
    return (
        weight * neighbor[0] + inv * center[0],
        weight * neighbor[1] + inv * center[1],
        weight * neighbor[2] + inv * center[2],
    )


class ImagePair:
    """A low quality input image paired with its clean upscaled target."""

    def __init__(self, inimg, outimg, scale=3):
        if scale != 3:
            raise NotImplementedError(
                "The network's architecture assumes a fixed 3x scale "
                "(a 3x3 input neighborhood produces a 3x3 output block); "
                "other scales require reworking the output block sizing."
            )
        self.inimg = inimg
        self.outimg = outimg
        self.scale = scale

    @classmethod
    def load(cls, input_path, output_path, scale=3):
        return cls(imageio.imread(input_path), imageio.imread(output_path), scale=scale)

    def _input_neighborhood(self, x, y):
        im = self.inimg
        center = im[y][x]
        v = [
            _blend(im[y - 1][x - 1], center, CORNER_WEIGHT), _blend(im[y - 1][x], center, EDGE_WEIGHT), _blend(im[y - 1][x + 1], center, CORNER_WEIGHT),
            _blend(im[y][x - 1], center, EDGE_WEIGHT),        pixel_to_3(center),                         _blend(im[y][x + 1], center, EDGE_WEIGHT),
            _blend(im[y + 1][x - 1], center, CORNER_WEIGHT), _blend(im[y + 1][x], center, EDGE_WEIGHT), _blend(im[y + 1][x + 1], center, CORNER_WEIGHT),
        ]
        return [pixel_to_3(p) for p in v]

    def _output_block(self, x, y):
        im = self.outimg
        ox, oy = self.scale * x, self.scale * y
        v = [
            im[oy - 1][ox - 1], im[oy - 1][ox], im[oy - 1][ox + 1],
            im[oy][ox - 1],     im[oy][ox],     im[oy][ox + 1],
            im[oy + 1][ox - 1], im[oy + 1][ox], im[oy + 1][ox + 1],
        ]
        return [pixel_to_3(p) for p in v]

    def input_patch(self, x, y):
        """Normalized 3x3 input neighborhood centered on (x, y)."""
        return [normalize(p) for p in self._input_neighborhood(x, y)]

    def output_patch(self, x, y):
        """Normalized 3x3 target block, the clean upscaled counterpart of (x, y)."""
        return [normalize(p) for p in self._output_block(x, y)]
