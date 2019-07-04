import pytest

import fullsatsolver as fss


def test_NCKProblemEncoder():
    pe = fss.NCKProblemEncoder(10)
    assert pe(2, [2,4,6]) == [
        [11, 2], [4, 12], [-11, 12], [4, -11], [6, -12],
        [-2, -4, -6],
    ]
    assert pe(1, [1,6,7,8]) == [
        [1, 6, 7, 8],
        [13, -1],
        [-6, 14],
        [-13, 14],
        [-6, -13],
        [-7, 15],
        [-14, 15],
        [-7, -14],
        [-8, -15],
    ]
