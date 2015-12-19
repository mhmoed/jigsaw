"""
This file contains general functionality not directly related to the LP-based
algorithm, such as image manipulation and IO functionality.
"""
import math
from itertools import product

import numpy as np
import skimage.io as io
from linprog import solve as solve_lp


def subdivide_image(image, num_pieces):
    """
    Subdivide an image into the given number of pieces. The image must be
    square, and its dimensions divisible by the number of pieces.

    :param image: image to subdivide.
    :param num_pieces: number of pieces to subdivide image into.
    :return: list of subimages.
    """
    h, w, _ = image.shape
    if h != w:
        raise ValueError('image is not square')
    num_blocks_per_dimension = int(math.sqrt(num_pieces))
    if h % num_blocks_per_dimension:
        raise ValueError('block size is not an integer')
    block_size = h / num_blocks_per_dimension
    block_itr = product(range(0, h, block_size), range(0, h, block_size))
    return [image[y:y + block_size, x:x + block_size] for y, x in block_itr]


def load_image(filename):
    """
    Load image and strip any colour channels other than RGB.

    This method uses scikit-images's io functionality under the hood.

    :param filename: filename of file to load.
    :return: loaded image.
    """
    image = io.imread(filename)
    _, _, num_colour_channels = image.shape
    if num_colour_channels > 3:
        return image[:, :, :3]
    return image


def save_image(filename, array):
    """
    Save image to a file.

    :param filename: filename of output file.
    :param array: numpy array to save as an image.
    """
    io.imsave(filename, array)


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

    xs = np.array(x, dtype=np.int32)
    ys = np.array(y, dtype=np.int32)

    for image, (x, y) in zip(images, zip(xs, ys)):
        sx, sy, dx, dy = x * dimension, y * dimension, \
                         (x + 1) * dimension, (y + 1) * dimension
        canvas[sy:dy, sx:dx] = image
    return canvas
