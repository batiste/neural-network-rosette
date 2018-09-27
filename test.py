import numpy as np
from images import values_in, values_out, inimg, outimg
import imageio

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], 26)
        print self.weights1
        self.weights2   = np.random.rand(26, 3)
        print self.weights2
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

def convert(p):
    return (255 * p[0], 255 * p[1], 255 * p[2])

def storePixels(array, index, pixels, width):
    array[index - width - 1] = convert(pixels[0])
    array[index - width] = convert(pixels[1])
    array[index - width + 1] = convert(pixels[2])

    array[index - 1] = convert(pixels[3])
    array[index] = convert(pixels[4])
    array[index + 1] = convert(pixels[5])

    array[index + width - 1] = convert(pixels[6])
    array[index + width] = convert(pixels[7])
    array[index + width + 1] = convert(pixels[8])


if __name__ == "__main__":
    _x = np.random.randint(5, 120)
    _y = np.random.randint(5, 50)
    print _x, _y
    input = values_in(_x, _y)
    output = values_out(_x, _y)

    pixels = []
    y = 1
    x = 1
    for pixel in input:
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))

    for pixel in output:
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))

    #
    X = np.array(input)
    y = np.array(output)
    nn = NeuralNetwork(X, y)

    for z in range(10000):
        x = np.random.randint(2, 123)
        y = np.random.randint(2, 55)
        input = values_in(x, y)
        output = values_out(x, y)
        nn.input = np.array(input)
        nn.y = np.array(output)
        nn.feedforward()
        nn.backprop()
    #
    print "Neural network trained"


    for pixel in nn.feedforward():
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))

    # another random pixel
    _x = np.random.randint(5, 120)
    _y = np.random.randint(5, 50)
    print _x, _y
    input = values_in(_x, _y)
    output = values_out(_x, _y)

    for pixel in input:
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))
    for pixel in output:
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))

    nn.input = input
    for pixel in nn.feedforward():
        pixels.append((255 * pixel[0], 255 * pixel[1], 255 * pixel[2]))


    a = np.array(pixels, dtype=np.uint8)
    a.shape = (6, 9, 3)
    imageio.imwrite('hop.png', a)
    print "Image written"

    # print nn.weights1
    # print nn.weights2
    # print "---"
    #
    [height, width, _] = inimg.shape
    # 57 124, 6710
    [oheight, owidth, _] = outimg.shape
    outw = owidth - 6
    outh = oheight - 6
    assert (width - 2) * 3 == outw
    assert (height - 2) * 3 == outh
    pixels = [None] * (outh * outw)
    print len(pixels)
    y = 1
    x = 1
    while(y < height - 1):
        while(x < width - 1):
            nn.input = values_in(x, y)
            index = (outw + (3 * outw * (y - 1))) + (1 + ((x - 1) * 3))
            storePixels(pixels, index, nn.feedforward(), outw)
            # print x, y, index, len(pixels)
            x += 1
        else:
            x = 1
        y += 1

    a = np.array(pixels, dtype=np.uint8)
    a.shape = (outh, outw, 3)
    imageio.imwrite('all.png', a)
    print "Image written"