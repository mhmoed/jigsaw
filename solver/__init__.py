"""
Implement jigsaw puzzle solver from Yu et al. (2015).

References:
    Yu, R., Russell, C., & Agapito, L. (2015). Solving Jigsaw Puzzles with
    Linear Programming. arXiv preprint arXiv:1511.04472.
    Gallagher, A. C. (2012). Jigsaw puzzles with pieces of unknown
    orientation. In Computer Vision and Pattern Recognition (CVPR), 2012 IEEE
    Conference on (pp. 382-389). IEEE.
"""
from itertools import product, groupby
import random

import skimage.io as io
import numpy as np

from itertools import chain

from scipy.spatial.distance import mahalanobis
from scipy.optimize import linprog

import matplotlib.pyplot as plt


NUM_CONSTRAINTS = 2
MATCH_REJECTION_THRESHOLD = 10e-5
MGC_NUM_ROTATIONS = [3, 0, 1, 2]
NUM_ORIENTATIONS = len(MGC_NUM_ROTATIONS)
DELTA_X = [0, -1, 0, 1]
DELTA_Y = [1, 0, -1, 0]


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


def compute_weights(pairwise_matches, mgc_distances):
    """
    Compute weights for specified pairwise matches and their MGC distances.

    A weight w_ijo is defined according to Yu et al. as the inverse
    ratio between the MGC distance for w_ijo, D_ijo, and the best alternative
    match for all k =/= i and k =/= j, for this orientation. For details, see
    Yu et al. (2015), equation 1.

    :param pairwise_matches: list of (image index 1, image index 2, orientation)
      tuples.
    :param mgc_distances: dictionary of (image index 1, image index 2,
    orientation) -> MGC distance items.
    :return: dictionary of weights, with tuples identical to those in
    pairwise_matches, and their weights as values.
    """
    num_images = max((i for i, _, _ in pairwise_matches)) + 1
    index_set = frozenset(range(num_images))
    weights = {}
    for i, j, o in pairwise_matches:
        min_row = min(mgc_distances[k, j, o] for k in index_set - {i})
        min_col = min(mgc_distances[i, k, o] for k in index_set - {j})
        weights[i, j, o] = min(min_row, min_col) / mgc_distances[i, j, o]
    return weights


def compute_active_selection(pairwise_matches, mgc_distances):
    """
    Select a subset of best matches from the specified list of pairwise matches.

    A best match for a given (i, j, o) in the list of pairwise matches, is
    defined as that (i, j, o) tuple that minimises the MGC distance for that j.
    In other words, for all matches in the list of pairwise matches, take that
    match for which the match with this orientation minimises the MGC distance.

    For details, see Yu et al. (2015), equation 14.

    :param pairwise_matches: list of (image index 1, image index 2, orientation)
     tuples.
    :param mgc_distances: dictionary of (image index 1, image index 2,
    orientation) -> MGC distance items.
    :return:
    """
    def i_o_key((i, _, o)):
        return i, o

    active_selection = []
    for _, group in groupby(sorted(pairwise_matches, key=i_o_key), i_o_key):
        entries = list(group)
        distances = np.array([mgc_distances[entry] for entry in entries])
        lowest_index = np.argmin(distances)
        entry = entries[lowest_index]
        active_selection.append(entry)
    return active_selection


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
                        range(NUM_ORIENTATIONS)))


def reconstruct(images, x, y):
    dimension = images[0].shape[0]
    num_blocks = int(np.sqrt(len(images)))
    canvas_size = num_blocks * dimension
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    for image, (x, y) in zip(images, zip(x.astype(np.int32), y.astype(np.int32))):
        sx, sy, dx, dy = x * dimension, y * dimension, (x + 1) * dimension, (y + 1) * dimension
        canvas[sy:dy, sx:dx] = image
    return canvas


def compute_solution(active_selection, weights, maxiter=None):
    def sorted_by_i_and_o(active_selection):
        return sorted(active_selection, key=lambda (i, _, o): (i, o))

    def row_index(i, o):
        return 8 * i + NUM_CONSTRAINTS * o

    # Sort a by i and o. The resulting order allows for simplifications on A_ub

    n = len(active_selection) / NUM_ORIENTATIONS
    sorted_a = sorted_by_i_and_o(active_selection)

    # Construct A_ub, given as follows:
    #    A_ub = | H1 | 0  | X |
    #           | 0  | H2 | Y |,
    # and where X = Y and H1 = H2 (constraints are identical for X and Y).

    h_base = np.array([-1] * NUM_CONSTRAINTS + [0] * (NUM_ORIENTATIONS * NUM_CONSTRAINTS * n - NUM_CONSTRAINTS))
    H = np.array([np.roll(h_base, k) for k in range(0, NUM_ORIENTATIONS * NUM_CONSTRAINTS * n, NUM_CONSTRAINTS)]).T

    xi_base = np.array([1, -1] * NUM_ORIENTATIONS + [0] * (NUM_ORIENTATIONS * NUM_CONSTRAINTS) * (n - 1))
    Xi = np.array([np.roll(xi_base, k) for k in
                   range(0, NUM_ORIENTATIONS * NUM_CONSTRAINTS * n, NUM_CONSTRAINTS * NUM_ORIENTATIONS)]).T

    Xj = np.zeros(Xi.shape, dtype=np.int32)
    for i, j, o in sorted_a:
        r = row_index(i, o)
        Xj[r:r + 2, j] = [-1, 1]
    X = Xi + Xj

    # Construct A_ub

    h, w = H.shape
    Z_h = np.zeros((h, w), dtype=np.int32)
    Z_x = np.zeros((h, n), dtype=np.int32)
    A_ub = np.vstack([H, Z_h])
    A_ub = np.hstack([A_ub, np.vstack([Z_h, H])])
    A_ub = np.hstack([A_ub, np.vstack([X, Z_x])])
    A_ub = np.hstack([A_ub, np.vstack([Z_x, X])])

    # Construct b_ub

    b_x = list(chain.from_iterable([[DELTA_X[o], -DELTA_X[o]] for (_, _, o) in active_selection]))
    b_y = list(chain.from_iterable([[DELTA_Y[o], -DELTA_Y[o]] for (_, _, o) in active_selection]))
    b_ub = np.array(b_x + b_y)

    # Construct c

    c_base = [weights[_] for _ in active_selection]
    c = np.array(c_base * 2 + ([0] * 2 * n))

    # Calculate solution

    options = {'maxiter': maxiter} if maxiter else {}
    solution = linprog(c, A_ub, b_ub, options=options)

    if not solution.success:
        if solution.message == 'Iteration limit reached.':
            raise ValueError('iteration limit reached, try increasing the number of max iterations')
        else:
            raise ValueError('unable to find solution to LP: {}'.format(solution.message))

    xy = solution.x[-n * 2:]
    return xy[:n], xy[n:]


def get_rejected_matches(active_selection, x, y):
    rejected_matches = set()
    for i, j, o in active_selection:
        if abs(x[i] - x[j] - DELTA_X[o]) > MATCH_REJECTION_THRESHOLD:
            rejected_matches.add((i, j, o))
        if abs(y[i] - y[j] - DELTA_Y[o]) > MATCH_REJECTION_THRESHOLD:
            rejected_matches.add((i, j, o))
    return rejected_matches


def solve(images, maxiter=None):
    # Initialise to A^0, U^0 and x^0, y^0

    pairwise_matches = initial_pairwise_matches(len(images))
    mgc_distances = compute_mgc_distances(images, pairwise_matches)
    weights = compute_weights(pairwise_matches, mgc_distances)
    active_selection = compute_active_selection(pairwise_matches, mgc_distances)
    x, y = compute_solution(active_selection, weights, maxiter)

    # Iterate until converged

    old_x, old_y = None, None

    while (old_x is None and old_y is None) or not (np.array_equal(old_x, x) and np.array_equal(old_y, y)):
        rejected_matches = get_rejected_matches(active_selection, x, y)
        pairwise_matches = list(set(pairwise_matches) - rejected_matches)
        active_selection = compute_active_selection(pairwise_matches, mgc_distances)

        old_x, old_y = x, y
        x, y = compute_solution(active_selection, weights, maxiter)

    return x, y


def main():
    shuffled = io.imread('data/shuffled.bmp')

    # Subdivide into blocks

    IMAGE_SIZE = 800
    BLOCK_SIZE = 200
    NUM = IMAGE_SIZE / BLOCK_SIZE

    block_itr = product(range(0, IMAGE_SIZE, BLOCK_SIZE), range(0, IMAGE_SIZE, BLOCK_SIZE))

    images = [shuffled[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] for y, x in block_itr]
    random.shuffle(images)

    x, y = solve(images, maxiter=15000)
    plt.imsave('/tmp/out.png', reconstruct(images, x, y))


if __name__ == '__main__':
    main()
