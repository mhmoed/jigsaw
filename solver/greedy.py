"""
Solves a jigsaw puzzle using greedy matching based on Mahalanobis Gradient
Compatibility (MGC) distances between pieces.

Pairwise distances between pieces are calculated for all four orientations
using MGC. The algorithm is initialised with a random piece. Then, the best
matching piece is selected for all open tiles next to all existing pieces, and
placed accordingly. This process is repeated until no pieces are left to assign.
"""
import random
from itertools import product

import numpy as np

import solver as slv


def solve(images, random_seed=None):
    """
    Solve a jigsaw puzzle greedily.

    :param images: a list of images.
    :param random_seed: random seed to initialise with, or None if default.
    :return: an (x, y) tuple, where x and y are both of length |images|, with
    the x and y coordinates of each image.
    """
    if random_seed:
        random.seed(random_seed)

    num_images = len(images)

    # Pre-calculate MGC distances between all pieces for all orientations

    distances = slv.compute_mgc_distances(images, slv.initial_pairwise_matches(
        num_images))

    # Create canvas

    dimension = num_images * 2
    canvas = np.ones((dimension, dimension), dtype=np.int32) * -1

    # Initialise with a random piece in the middle of the canvas

    indices = list(range(num_images))
    index = random.choice(indices)
    canvas[dimension / 2, dimension / 2] = index

    # Iterate until no pieces are left to place

    used_indices = {index}
    indices.remove(index)

    while indices:
        # Enumerate all open boundaries

        boundaries = compute_open_boundaries(canvas)

        # Match all pieces j to all boundaries (x, y, o)

        min_match, min_score = -1, np.inf
        for x, y, o, i in boundaries:
            for j in set(range(num_images)) - used_indices:
                score = distances[i, j, o]
                if score < min_score:
                    min_score = score
                    min_match = (x, y, o, j, i)

        # Add best matching piece

        x, y, o, j, i = min_match
        if o == 0:
            canvas[y - 1, x] = j
        elif o == 1:
            canvas[y, x + 1] = j
        elif o == 2:
            canvas[y + 1, x] = j
        else:
            canvas[y, x - 1] = j

        # Remove index of piece just placed so it won't get added again

        indices.remove(j)
        used_indices.add(j)

    # Trim canvas

    return canvas_to_xy(trimmed_canvas(canvas))


def trimmed_canvas(canvas):
    """
    Trim a canvas to its smallest form. In most cases, rows and columns of the
    canvas will be unused (i.e. filled with only -1's). These rows and columns
    are removed, until a solution is left with pieces arranged contiguously.
    :param canvas:
    :return:
    """
    dimension, _ = canvas.shape
    mask = canvas == -1
    row_filter = list(np.all(mask, axis=1))
    column_filter = list(np.all(mask, axis=0))
    sy = row_filter.index(False)
    dy = dimension - list(reversed(row_filter)).index(False)

    sx = column_filter.index(False)
    dx = dimension - list(reversed(column_filter)).index(False)
    return canvas[sy:dy, sx:dx]


def canvas_to_xy(canvas):
    num_images = len(canvas) ** 2

    x, y = [0] * num_images, [0] * num_images
    for y_, row in enumerate(canvas):
        for x_, item in enumerate(row):
            x[item] = x_
            y[item] = y_

    return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def compute_open_boundaries(canvas):
    def filled(element):
        return element != -1

    def safe_access(x, y):
        try:
            return canvas[y, x]
        except IndexError:
            return -1

    boundaries = []
    height, width = canvas.shape
    for x, y in product(range(width), range(height)):
        piece = canvas[y, x]
        if filled(piece):
            if not filled(safe_access(x, y - 1)):  # j on top of i
                boundaries.append((x, y, 0, piece))
            if not filled(safe_access(x + 1, y)):  # j to the right of i
                boundaries.append((x, y, 1, piece))
            if not filled(safe_access(x, y + 1)):  # i on top of j
                boundaries.append((x, y, 2, piece))
            if not filled(safe_access(x - 1, y)):  # j to the left of i
                boundaries.append((x, y, 3, piece))
    return boundaries
