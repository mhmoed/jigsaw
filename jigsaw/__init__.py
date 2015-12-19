from itertools import product
import math

import skimage.io as io
import numpy as np

from linprog import solve as solve_lp
from scipy.spatial.distance import mahalanobis

MGC_NUM_ROTATIONS = [3, 0, 1, 2]
MGC_NUM_ORIENTATIONS = len(MGC_NUM_ROTATIONS)


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
    image = io.imread(filename)
    _, _, num_colour_channels = image.shape
    if num_colour_channels > 3:
        return image[:, :, :3]
    return image


def save_image(filename, array):
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


def compute_mgc_distances(images, pairwise_matches):
    """
    Compute MGC distances for all specified images and their pairwise matches.

    :param images: list of images.
    :param pairwise_matches: list of (image index 1, image index 2, orientation)
     tuples.
    :return: dictionary with tuples from pairwise_matches as keys, and their
    resulting MGCs as values.
    """
    return {(i, j, o): mgc(images[i], images[j], o) for
            i, j, o in pairwise_matches}


def mgc(image1, image2, orientation):
    """
    Calculate the Mahalanobis Gradient Compatibility (MGC) of image 1 relative
    to image 2. MGC provides a measure of the similarity in gradient
    distributions between the boundaries of adjoining images with respect to
    a particular orientation. For detailed information on the underlying,
    please see Gallagher et al. (2012).

    Orientations are integers, defined according to Yu et al. (2015):
    - 0: measure MGC between the top of image 1 and bottom of image 2;
    - 1: measure MGC between the right of image 1 and left of image 2;
    - 2: measure MGC between the bottom of image 1 and top of image 2;
    - 3: measure MGC between the left of image 1 and right of image 2;

    Both images are first rotated into position according to the specified
    orientations, such that the right side of image 1 and the left side of
    image 2 are the boundaries of interest. This preprocessing step simplifies
    the subsequent calculation of the MGC, but increases computation time.
    Therefore, a straightforward optimisation would be to extract boundary
    sequences directly.

    NOTE: nomenclature taken from Gallagher et al. (2012).

    :param orientation: orientation image 1 relative to image 2.
    :param image1: first image.
    :param image2: second image.
    :return MGC.
    """
    assert image1.shape == image2.shape, 'images must be of same dimensions'
    assert orientation in MGC_NUM_ROTATIONS, 'invalid orientation'

    num_rotations = MGC_NUM_ROTATIONS[orientation]

    # Rotate images based on orientation - this is easier than extracting
    # the sequences based on an orientation case switch

    image1_signed = np.rot90(image1, num_rotations).astype(np.int16)
    image2_signed = np.rot90(image2, num_rotations).astype(np.int16)

    # Get mean gradient of image1

    g_i_l = image1_signed[:, -1] - image1_signed[:, -2]
    mu = g_i_l.mean(axis=0)

    # Get covariance matrix S
    # Small values are added to the diagonal of S to resolve non-invertibility
    # of S. This will not influence the final result.

    s = np.cov(g_i_l.T) + np.eye(3) * 10e-6

    # Get G_ij_LR

    g_ij_lr = image2_signed[:, 1] - image1_signed[:, -1]

    return sum(mahalanobis(row, mu, np.linalg.inv(s)) for row in g_ij_lr)


def initial_pairwise_matches(num_images):
    """
    Calculate initial pairwise matches for a given number of images. Initial
    pairwise matches are all possible combinations of image 1, image 2, and
    orientation. Given that the number of orientations is 4, and matches are
    pairwise, the number of pairwise matches always equals 4 * n^2, where n
    equals the number of images.

    :param num_images: number of images in puzzle.
    :return: initial pairwise matches, as a list of (image index 1, image index
    2, orientation) tuples.
    """
    return list(product(range(num_images), range(num_images),
                        range(MGC_NUM_ORIENTATIONS)))
