import typing
import time
import sys
import logging
import itertools

import numpy as np
from scipy.spatial import distance as sp_dist
import pyautogui as pg

import actionplanner as planner


# pylint: disable=too-few-public-methods
class MouseClicker(planner.MouseClicker):
    def __init__(self, bdetector):
        """
        :type bdetector: virtual.vboard.BoardDetector
        """
        super().__init__(None, None, bdetector)

    def do_click(self, ploc: typing.Tuple[int, int], leftbutton: bool):
        if leftbutton:
            self.bd.left_click_cell(ploc)
        else:
            self.bd.flag_cell(ploc)
        time.sleep(pg.PAUSE)
        if sys.platform == 'darwin':
            time.sleep(pg.DARWIN_CATCH_UP_TIME)


class LBMouseClicker(MouseClicker):
    """
    MouseClicker that buffers left clicks till commit.
    """
    def __init__(self, bdetector):
        """
        :param mon: ...
        :param dpr: ...
        :param bdetector: the ``BoardDetector`` to use
        :param sct: an ``mss.mss`` instance
        """
        super().__init__(bdetector)
        self.left_bx = np.array([], dtype=int)
        self.left_by = np.array([], dtype=int)

    def click(self, blocs, leftbutton):
        bx, by = blocs
        if isinstance(leftbutton, bool):
            leftbutton = np.array(list(itertools.repeat(leftbutton, len(bx))))
        right_blocs = bx[~leftbutton], by[~leftbutton]
        if np.any(~leftbutton):
            self._l.info('right clicks: %s', list(zip(*right_blocs)))
        for pxy in zip(*self.bd.boardloc_as_pixelloc(right_blocs)):
            self.do_click(pxy, False)
        self.left_bx = np.append(self.left_bx, bx[leftbutton])
        self.left_by = np.append(self.left_by, by[leftbutton])

    def commit(self):
        if self.left_bx.shape[0]:
            blocs = self.left_bx, self.left_by
            planner.buffered_homo_clicks(self.bd, None, blocs, True,
                                         self.do_click, self._l)
        self.left_bx = np.array([], dtype=int)
        self.left_by = np.array([], dtype=int)


class NatChrfBMouseClicker(MouseClicker):
    """
    ``NatChrfBMouseClicker`` using Christofides algorithm to reorder buffered
    clicks with natural mouse movement.
    """
    def __init__(self, bdetector):
        super().__init__(bdetector)
        self.prev_ploc = None
        self.unit_dur = 0.07
        self.left_bx = np.array([], dtype=int)
        self.left_by = np.array([], dtype=int)
        self.right_bx = np.array([], dtype=int)
        self.right_by = np.array([], dtype=int)

    def do_click(self, ploc: typing.Tuple[int, int], leftbutton: bool):
        super().do_click(ploc, leftbutton)
        if self.prev_ploc is not None:
            pd = sp_dist.euclidean(ploc, self.prev_ploc)
            self._l.info('mouse cursor move distance: %f', pd)
            dur = self.unit_dur * pd
        else:
            dur = 0.0
        time.sleep(dur)
        self.prev_ploc = ploc

    def click(self, blocs, leftbutton):
        bx, by = blocs
        if isinstance(leftbutton, bool):
            leftbutton = np.array(list(itertools.repeat(leftbutton, len(bx))))
        self.left_bx = np.append(self.left_bx, bx[leftbutton])
        self.left_by = np.append(self.left_by, by[leftbutton])
        self.right_bx = np.append(self.right_bx, bx[~leftbutton])
        self.right_by = np.append(self.right_by, by[~leftbutton])

    def _commit_button(self, leftbutton: bool):
        if leftbutton:
            bx, by = self.left_bx, self.left_by
        else:
            bx, by = self.right_bx, self.right_by
        if bx.shape[0] > 1:
            blocs = planner.christofide_reorder(self.bd, bx, by,
                                                self.prev_ploc)
        else:
            blocs = bx, by
        if bx.shape[0] > 0:
            planner.buffered_homo_clicks(self.bd, None, blocs, leftbutton,
                                         self.do_click, self._l)
        if leftbutton:
            self.left_bx = np.array([], dtype=int)
            self.left_by = np.array([], dtype=int)
        else:
            self.right_bx = np.array([], dtype=int)
            self.right_by = np.array([], dtype=int)

    def commit(self):
        # this order is important
        self._commit_button(False)
        self._commit_button(True)
