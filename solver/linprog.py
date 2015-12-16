"""
Implement jigsaw puzzle solver from Yu et al. (2015).

References:
    Yu, R., Russell, C., & Agapito, L. (2015). Solving Jigsaw Puzzles with
    Linear Programming. arXiv preprint arXiv:1511.04472.
    Gallagher, A. C. (2012). Jigsaw puzzles with pieces of unknown
    orientation. In Computer Vision and Pattern Recognition (CVPR), 2012 IEEE
    Conference on (pp. 382-389). IEEE.
"""
from itertools import chain, groupby

import numpy as np
from scipy.optimize import linprog

import solver as slv

NUM_CONSTRAINTS = 2
MATCH_REJECTION_THRESHOLD = 10e-5
DELTA_X = [0, -1, 0, 1]
DELTA_Y = [1, 0, -1, 0]


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


def compute_solution(active_selection, weights, maxiter=None):
    """
    Compute the solution of a linear program (LP) given by the active selection
    and weights lookup table.

    An LP is constructed based on the current active selection and weights
    lookup table. The objective function to be minimised is the sum of the
    number of mismatches in the x direction, plus the number of mismatches in
    the y direction. Mismatches are multiplied by their MGC distance.

    For a detailed overview, see Yu et al. (2015), equation 16.

    :param active_selection: list of (image index 1, image index 2,
    orientation) tuples representing the current active selection.
    :param weights: weights lookup table, as computed by compute_weights.
    :param maxiter: maximum number of iterations of the simplex method.
    :return: (x, y) tuple with x and y coordinates.
    """
    def sorted_by_i_and_o(active_selection):
        return sorted(active_selection, key=lambda (i, _, o): (i, o))

    def row_index(i, o):
        return 8 * i + NUM_CONSTRAINTS * o

    # Sort active selection by i and o. The resulting order allows for
    # simplifications on the upper bound matrix A_ub.

    n = len(active_selection) / slv.MGC_NUM_ORIENTATIONS
    sorted_a = sorted_by_i_and_o(active_selection)

    # Construct A_ub, given as follows:
    #    A_ub = | H1 | 0  | X |
    #           | 0  | H2 | Y |,
    # and where X = Y and H1 = H2 (constraints are identical for X and Y).

    h_base = np.array([-1] * NUM_CONSTRAINTS + [0] * (slv.MGC_NUM_ORIENTATIONS *
                                                      NUM_CONSTRAINTS *
                                                      n - NUM_CONSTRAINTS))
    H = np.array([np.roll(h_base, k) for k in range(0,
                                                    slv.MGC_NUM_ORIENTATIONS *
                                                    NUM_CONSTRAINTS * n,
                                                    NUM_CONSTRAINTS)]).T

    xi_base = np.array([1, -1] * slv.MGC_NUM_ORIENTATIONS + [0] *
                       (slv.MGC_NUM_ORIENTATIONS * NUM_CONSTRAINTS) * (n - 1))
    Xi = np.array([np.roll(xi_base, k) for k in
                   range(0, slv.MGC_NUM_ORIENTATIONS * NUM_CONSTRAINTS * n,
                         NUM_CONSTRAINTS * slv.MGC_NUM_ORIENTATIONS)]).T

    Xj = np.zeros(Xi.shape, dtype=np.int32)
    for i, j, o in sorted_a:
        r = row_index(i, o)
        Xj[r:r + 2, j] = [-1, 1]
    X = Xi + Xj

    # Construct A_ub by vertically and horizontally stacking its constituent
    # matrices. Although pre-allocating the matrix and copying values may be
    # more efficient, it makes for less readable code.

    h, w = H.shape
    Z_h = np.zeros((h, w), dtype=np.int32)
    Z_x = np.zeros((h, n), dtype=np.int32)
    A_ub = np.vstack([H, Z_h])
    A_ub = np.hstack([A_ub, np.vstack([Z_h, H])])
    A_ub = np.hstack([A_ub, np.vstack([X, Z_x])])
    A_ub = np.hstack([A_ub, np.vstack([Z_x, X])])

    # Construct b_ub

    b_x = list(chain.from_iterable([[DELTA_X[o], -DELTA_X[o]]
                                    for (_, _, o) in sorted_a]))
    b_y = list(chain.from_iterable([[DELTA_Y[o], -DELTA_Y[o]]
                                    for (_, _, o) in sorted_a]))
    b_ub = np.array(b_x + b_y)

    # Construct c

    c_base = [weights[_] for _ in active_selection]
    c = np.array(c_base * 2 + ([0] * 2 * n))

    # Calculate solution

    options = {'maxiter': maxiter} if maxiter else {}
    solution = linprog(c, A_ub, b_ub, options=options)

    if not solution.success:
        if solution.message == 'Iteration limit reached.':
            raise ValueError('iteration limit reached, try increasing the ' +
                             'number of max iterations')
        else:
            raise ValueError('unable to find solution to LP: {}'.format(
                solution.message))

    xy = solution.x[-n * 2:]
    return xy[:n], xy[n:]


def compute_rejected_matches(active_selection, x, y):
    rejected_matches = set()
    for i, j, o in active_selection:
        if abs(x[i] - x[j] - DELTA_X[o]) > MATCH_REJECTION_THRESHOLD:
            rejected_matches.add((i, j, o))
        if abs(y[i] - y[j] - DELTA_Y[o]) > MATCH_REJECTION_THRESHOLD:
            rejected_matches.add((i, j, o))
    return rejected_matches


def solve(images, maxiter=None):
    # Initialise to A^0, U^0 and x^0, y^0

    pairwise_matches = slv.initial_pairwise_matches(len(images))
    mgc_distances = slv.compute_mgc_distances(images, pairwise_matches)
    weights = compute_weights(pairwise_matches, mgc_distances)
    active_selection = compute_active_selection(pairwise_matches, mgc_distances)
    x, y = compute_solution(active_selection, weights, maxiter)

    # Iterate until converged

    old_x, old_y = None, None

    while (old_x is None and old_y is None) or not (np.array_equal(old_x, x) and np.array_equal(old_y, y)):
        rejected_matches = compute_rejected_matches(active_selection, x, y)
        pairwise_matches = list(set(pairwise_matches) - rejected_matches)
        active_selection = compute_active_selection(pairwise_matches, mgc_distances)

        old_x, old_y = x, y
        x, y = compute_solution(active_selection, weights, maxiter)

    return x, y
