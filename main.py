import numpy as np
from images import values_in, values_out, inimg, outimg, normalized_values
import imageio
import sys

# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        # 26 seems the maximun of node in the layer 1 before everything
        # goes crazy
        self.weights1   = np.random.rand(self.input.shape[1], 18)
        self.weights2   = np.random.rand(18, 3)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,
            (np.dot(2*(self.y - self.output)
            * sigmoid_derivative(self.output), self.weights2.T)
            * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

def convert(p):
    return (255 * p[0], 255 * p[1], 255 * p[2])

def writeImage(inimg, outimg, nn, filename):
    [height, width, _] = inimg.shape
    [oheight, owidth, _] = outimg.shape
    outw = owidth - 6
    outh = oheight - 6
    assert (width - 2) * 3 == outw
    assert (height - 2) * 3 == outh
    pixels = [None] * (outh * outw)
    y = 1
    x = 1
    while(y < height - 1):
        while(x < width - 1):
            nn.input = normalized_values(inimg, x, y)
            index = (outw + (3 * outw * (y - 1))) + (1 + ((x - 1) * 3))
            storePixels(pixels, index, nn.feedforward(), outw)
            x += 1
        else:
            x = 1
        y += 1

    a = np.array(pixels, dtype=np.uint8)
    a.shape = (outh, outw, 3)
    imageio.imwrite(filename, a)

def storePixels(array, index, pixels, width):
    # It works properly

    array[index - width - 1] = convert(pixels[0])
    array[index - width] = convert(pixels[1])
    array[index - width + 1] = convert(pixels[2])

    array[index - 1] = convert(pixels[3])
    array[index] = convert(pixels[4])
    array[index + 1] = convert(pixels[5])

    array[index + width - 1] = convert(pixels[6])
    array[index + width] = convert(pixels[7])
    array[index + width + 1] = convert(pixels[8])


def main(inimg, outimg):
    [height, width, _] = inimg.shape
    _x = np.random.randint(5, width - 2)
    _y = np.random.randint(5, height - 2)
    input = values_in(_x, _y)
    output = values_out(_x, _y)

    pixels = []
    y = 1
    x = 1
    for pixel in input:
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))

    for pixel in output:
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))

    X = np.array(input)
    y = np.array(output)
    nn = NeuralNetwork(X, y)

    print "Training neural network ..."
    # this speed up the training significantly
    pixelsCache = {}
    for z in range(60000):
        # train the network with random pixel from the source image
        if z % 5000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        x = np.random.randint(2, width - 2)
        y = np.random.randint(2, height - 2)
        key = "%d,%d" % (x, y)
        if not key in pixelsCache:
            pixelsCache[key] = (np.array(values_in(x, y)), np.array(values_out(x, y)))
        input, output = pixelsCache[key]
        nn.input = input
        nn.y = output
        nn.feedforward()
        nn.backprop()

    print ""
    print "Neural network trained"


    for pixel in nn.feedforward():
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))

    # another random pixel
    _x = np.random.randint(5, width - 2)
    _y = np.random.randint(5, height - 2)
    input = values_in(_x, _y)
    output = values_out(_x, _y)

    for pixel in input:
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))
    for pixel in output:
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))

    nn.input = input
    for pixel in nn.feedforward():
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))


    # write a test image with 6 lines of 9 pixels each
    # input
    # output
    # neural network output
    a = np.array(pixels, dtype=np.uint8)
    a.shape = (6, 9, 3)
    imageio.imwrite('check.png', a)
    print "Image written"


    print "Outputing the result images"
    writeImage(inimg, outimg, nn, 'result.png')

    inimg = imageio.imread('in.png')
    outimg = imageio.imread('out.png')
    writeImage(inimg, outimg, nn, 'result2.png')


if __name__ == "__main__":
    main(inimg, outimg)
