import numpy as np

import vboard


def test_detect_board():
    with np.load('_test_vboard.data.npz') as data:
        scr = data['desktop']
        expected = data['board']
    locator = vboard.detect_board(scr)
    board = locator.as_board(scr)
    assert np.all(board == expected)
