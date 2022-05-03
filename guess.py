"""
Guess strategies.
"""

import numpy as np


def global_uniform(board: np.ndarray):
    return np.unravel_index(np.random.randint(board.size), board.shape)


def prefer_edge(board: np.ndarray, all_blocs: np.ndarray,
                guess_edge_weight: float):
    """
    Random guess among all_blocs favoring edges by guess_edge_weight extent.

    :param board: the board
    :param all_blocs: of shape (m, 2) such that the ith row is the board
           coordinate of an empty cell
    :param guess_edge_weight: should be no less than 1.0
    :return: the chosen board coordinate of shape (2,)
    """
    on_edge = ((all_blocs[:, 0] == 0)
               | (all_blocs[:, 0] == board.shape[0] - 1)
               | (all_blocs[:, 1] == 0)
               | (all_blocs[:, 1] == board.shape[1] - 1))
    weights = np.where(on_edge, guess_edge_weight, 1.0)
    weights = weights / np.sum(weights)
    rand_bloc = all_blocs[np.random.choice(
        np.arange(all_blocs.shape[0]), p=weights)]
    return rand_bloc
