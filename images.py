import imageio
import numpy as np

inimg = imageio.imread('in.png')
outimg = imageio.imread('out.png')

def pixelTo3(p):
    return (p[0], p[1], p[2])

def normalize(p):
    return (p[0] / 255., p[1] / 255., p[2] / 255.)

def values(im, x, y):
    v = [
        im[y-1][x-1], im[y-1][x], im[y-1][x+1],
        im[y][x-1],   im[y][x],   im[y][x+1],
        im[y+1][x-1], im[y+1][x], im[y+1][x+1],
    ]
    return map(lambda b: pixelTo3(b), v)

def values_in(x, y):
    return map(lambda b: normalize(b), values(inimg, x, y))

def values_out(x, y):
    return map(lambda b: normalize(b), values(outimg, 3*x, 3*y))
