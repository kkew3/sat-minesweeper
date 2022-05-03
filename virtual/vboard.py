"""
A simulated board recognition module that provides the same interface as
vboard.py
"""

import itertools

import numpy as np
import networkx as nx

import solverutils as sutils
from solverutils import CID


class InvalidVirtualClickError(Exception):
    pass


class BoardDetector:
    def __init__(self, key_board, board_graph):
        self.board_graph = board_graph
        if np.any((key_board == CID['f']) | (key_board == CID['q'])):
            raise ValueError('this is not a key board: ' + str(key_board))
        self.key_board = key_board
        self.cur_board = np.empty(key_board.shape, dtype=int)
        self.cur_board.fill(CID['q'])
        self.iboxof = sutils.IBoxOf(self.cur_board.shape)

    @classmethod
    def new(cls, key_board):
        """
        :param key_board: the board with all cells uncovered
        """
        board_graph = nx.Graph()
        board_graph.add_nodes_from(
            itertools.product(
                range(key_board.shape[0]), range(key_board.shape[1])))
        iboxof = sutils.IBoxOf(key_board.shape)
        for bxy in zip(*np.nonzero(key_board == 0)):
            board_graph.add_edges_from(
                (bxy, p) for p in zip(*iboxof(bxy, exclude_center=True)))
        return cls(key_board, board_graph)

    # pylint: disable=no-self-use
    def boardloc_as_pixelloc(self, blocs):
        return blocs

    def recognize_board_and_mr(self, _sct):
        return np.copy(self.cur_board), None, None

    def left_click_cell(self, bloc):
        if self.cur_board[bloc] == CID['q']:
            if self.key_board[bloc] == 0:
                blocs = np.asarray(
                    list(nx.dfs_preorder_nodes(self.board_graph, bloc)))
                self.cur_board[blocs[:, 0], blocs[:, 1]] = \
                        self.key_board[blocs[:, 0], blocs[:, 1]]
            elif 1 <= self.key_board[bloc] <= 8:
                self.cur_board[bloc] = self.key_board[bloc]
            else:
                # self.key_board[bloc] == CID['m']
                self.cur_board[np.nonzero(
                    self.key_board == CID['m'])] = CID['m']
        elif self.cur_board[bloc] == 0:
            pass
        elif 1 <= self.cur_board[bloc] <= 8:
            box_blocs = self.iboxof(bloc)
            if np.sum(self.cur_board[box_blocs] == CID['f']) \
                    == self.cur_board[bloc] and \
                    np.any(self.cur_board[box_blocs] == CID['q']):
                for bxy in zip(*box_blocs):
                    if self.cur_board[bxy] == CID['q']:
                        self.left_click_cell(bxy)
        # else, self.cur_board[bloc] in (CID['f'], CID['m']), both cases do
        # nothing

    def flag_cell(self, bloc):
        if self.cur_board[bloc] == CID['q']:
            self.cur_board[bloc] = CID['f']
        elif self.cur_board[bloc] == CID['f']:
            raise InvalidVirtualClickError
        # else, self.cur_board[bloc] in (CID['m'], 0, 1, 2, ..., 8), do nothing


# pylint: disable=too-few-public-methods
class StageIdentifier:
    def __init__(self, key_board):
        self._key_board = key_board
        self.number_cells_count = np.sum((key_board >= 0) & (key_board <= 8))

    def identify_stage(self, _scr, board):
        cur_number_cells_count = np.sum((board >= 0) & (board <= 8))
        assert cur_number_cells_count <= self.number_cells_count, \
               (self._key_board, board)
        if cur_number_cells_count == self.number_cells_count:
            return 'win'
        if np.any(board == CID['m']):
            return 'lost'
        return 'ongoing'
