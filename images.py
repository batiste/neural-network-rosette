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

v = values(inimg, 50, 50)
a = np.array(v, dtype=np.uint8)
a.shape = (3, 3, 3)
imageio.imwrite('pixelin.png', a)

v = values(outimg, 150, 150)
a = np.array(v, dtype=np.uint8)
a.shape = (3, 3, 3)
imageio.imwrite('pixelout.png', a)

# pixels = []
# for line in im:
#     for pixel in line:
#         print pixel
#         pixels.append((pixel[0], pixel[1], pixel[2]))
# a = np.array(pixels, dtype=np.uint8)
# a.shape = (57, 124, 3)
# imageio.imwrite('all.png', a)

# for pixel in im:
#   print pixel
