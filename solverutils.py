"""
The solver interface is designed to read CSV input of the mine board and
write CSV output indicating which tiles to uncover and what shall be
expected underneath. It should have little integration with the vision and
the mouse action simulation modules.

This module also defines some utility function and exception found in solver
modules.
"""

import sys
import argparse
import fileinput
import itertools

import numpy as np

# This is the class ID of (f)lag, (m)ine, and (q)uery tile
CID = {
    'f': 9,
    'm': 10,
    'q': 11,
}


class NoSolutionError(Exception):
    pass


def boxof(array, center, radius=1):
    return array[max(0, center[0] - radius):center[0] + radius + 1,
                 max(0, center[1] - radius):center[1] + radius + 1]


# pylint: disable=too-few-public-methods
class IBoxOf:
    """
    When called, returns indicies of cells in the box of center.
    """
    def __init__(self, board_shape):
        # use the same buffer to speed up
        self.z = np.zeros(board_shape, dtype=np.bool_)

    def __call__(self, center, radius=1, exclude_center=False):
        self.z[max(0, center[0] - radius):center[0] + radius + 1,
               max(0, center[1] - radius):center[1] + radius + 1] = True
        if exclude_center:
            self.z[center] = False
        i = np.nonzero(self.z)
        # make self.z back to zeros
        self.z[i[0], i[1]] = False
        return i


def make_parser():
    parser = argparse.ArgumentParser(
        prog='python ' + sys.argv[0],
        description='Solves the mine board provided via stdin and gives the '
        'result via stdout. An optional number of mines '
        'remaining can be specified at the first line of the '
        'CSV file by `#mines N\'. An optional first board location to click '
        'can be specified at the second line of the CSV file by '
        '`#first_bloc M,N\'.')
    parser.add_argument(
        'board_csv',
        action='append',
        nargs='?',
        metavar='CSVFILE',
        help='the CSV file describing the board, or omitted '
        'to read from stdin')
    return parser


class EmptyCsvError(Exception):
    pass


def read_board(board_csv):
    mines_remain = None
    first_bloc = None
    with fileinput.input(board_csv if board_csv[0] else None) as infile:
        for line in infile:
            if not line.startswith('#'):
                break
            if line.startswith('#mines '):
                mines_remain = int(line[len('#mines '):].rstrip())
            elif line.startswith('#first_bloc '):
                first_bloc = line[len('#first_bloc '):].rstrip()
                bx, _, by = first_bloc.partition(',')
                first_bloc = int(bx), int(by)
        else:
            # either the csv file is empty, or it contains only `#'-starting
            # headers
            raise EmptyCsvError
        # `line` must have been defined, otherwise EmptyCsvError will have
        # been raised
        # pylint: disable=undefined-loop-variable
        infile = itertools.chain([line], infile)
        board = np.loadtxt(infile, delimiter=',', dtype=int)
    if not board.size:
        raise EmptyCsvError
    return board, mines_remain, first_bloc
