from itertools import product
import math

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

from linprog import solve as solve_lp


def subdivide_image(image, num_pieces):
    h, w, _ = image.shape
    if h != w:
        raise ValueError('image is not square')
    if h % num_pieces:
        raise ValueError('block size is not an integer')
    block_size = h / int(math.sqrt(num_pieces))
    block_itr = product(range(0, h, block_size), range(0, h, block_size))
    return [image[y:y + block_size, x:x + block_size] for y, x in block_itr]


def load_image(filename):
    return io.imread(filename)


def save_image(filename, array):
    plt.imsave(filename, array)


def reconstruct(images, x, y):
    """
    Reconstruct the entire image with a list of sub-images, and their x and y
    coordinates.

    :param images: list of sub-images.
    :param x: list of x coordinates of images.
    :param y: list of y coordinates of images.
    :return: reconstructed entire image.
    """
    dimension = images[0].shape[0]
    num_blocks = int(math.sqrt(len(images)))
    canvas_size = num_blocks * dimension
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    for image, (x, y) in zip(images, zip(x.astype(np.int32),
                                         y.astype(np.int32))):
        sx, sy, dx, dy = x * dimension, y * dimension, \
                         (x + 1) * dimension, (y + 1) * dimension
        canvas[sy:dy, sx:dx] = image
    return canvas
