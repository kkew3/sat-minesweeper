import pytest
import satinfer


def test_cnftemplatelib():
    tl = satinfer.CNFTemplateLib()
    assert tl.get([1,3,4], 2) == [
        [-1, -3, -4],
        [-1, 3, 4],
        [1, -3, 4],
        [1, 3],
        [1, 3, -4],
        [1, 3, 4],
        [1, 4],
        [3, 4],
    ]

def test_interprete_solutions():
    f = satinfer.interprete_solutions
    assert f([[1, -2, 3],
              [-1, 2, 3]])  == (True, [(3, True)])
    assert f([[1, -2, -3],
              [-1, -2, 3]]) == (True, [(2, False)])
    assert f([[1, -2, 3],
              [1, 2, -3],
              [-1, 2, 3],
              [-1, 2, -3]]) == (False, [(2, True)])
    assert f([[-1, -2, 3],
              [1, -2, -3],
              [-1, 2, 3],
              [-1, 2, -3]]) == (False, [(1, False)])
