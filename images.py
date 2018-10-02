import imageio
import numpy as np

inimg = imageio.imread('inkami.png')
outimg = imageio.imread('outkami.png')

def pixelTo3(p):
    return (p[0], p[1], p[2])

def normalize(p):
    return (p[0] / 255., p[1] / 255., p[2] / 255.)

m = 0.36
mi = 1 - m
def mean_2px(p1, p2):
    return (
        (m * p1[0] + mi * p2[0]),
        (m * p1[1] + mi * p2[1]),
        (m * p1[2] + mi * p2[2]),
    )

cm = 0.42
cmi = 1 - cm
def mean_2px_corner(p1, p2):
    return (
        (cm * p1[0] + cmi * p2[0]),
        (cm * p1[1] + cmi * p2[1]),
        (cm * p1[2] + cmi * p2[2]),
    )

def values(im, x, y):
    m = im[y][x]
    v = [
        mean_2px_corner(im[y-1][x-1], m),  mean_2px(im[y-1][x], m), mean_2px_corner(im[y-1][x+1], m),
        mean_2px(im[y][x-1], m),           m,                       mean_2px(im[y][x+1], m),
        mean_2px_corner(im[y+1][x-1], m),  mean_2px(im[y+1][x], m), mean_2px_corner(im[y+1][x+1], m),
    ]
    return map(lambda b: pixelTo3(b), v)

def _values_out(im, x, y):
    v = [
        im[y-1][x-1], im[y-1][x], im[y-1][x+1],
        im[y][x-1],   im[y][x],   im[y][x+1],
        im[y+1][x-1], im[y+1][x], im[y+1][x+1],
    ]
    return map(lambda b: pixelTo3(b), v)

def normalized_values(im, x, y):
    return map(lambda b: normalize(b), values(im, x, y))

def values_in(x, y, inimg=inimg):
    return map(lambda b: normalize(b), values(inimg, x, y))

def values_out(x, y):
    return map(lambda b: normalize(b), _values_out(outimg, 3*x, 3*y))
