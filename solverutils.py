"""
The solver interface is designed to read CSV input of the mine board and
write CSV output indicating which tiles to uncover and what shall be
expected underneath. It should have little integration with the vision and
the mouse action simulation modules.
"""

import argparse

import numpy as np


# This is the class ID of (f)lag, (m)ine, and (q)uery tile
CID = {
    'f': 9,
    'm': 10,
    'q': 11,
}


def make_parser():
    parser = argparse.ArgumentParser(
        description='Solves the mine board provided via stdin and gives the '
                    'result via stdout. An optional number of mines '
                    'remaining can be specified at the first line of the '
                    'CSV file by `#mines N\'.')
    parser.add_argument('board_csv', nargs='?', const=None, metavar='CSVFILE')
    return parser


def select_results(results, confidence):
    if np.sum(confidence > 1 - 1e-6) > 0:
        return np.where(confidence > 1 - 1e-6)
    return results[np.argmax(confidence)][np.newaxis]
