import typing
import time

import pyautogui as pg
from PIL import Image
import mss

import vboard as vb


def make_screenshot(sct):
    img = sct.grab(sct.monitors[1])
    img = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
    return img


# pylint: disable=too-few-public-methods
class MouseClicker:
    def __init__(self):
        with mss.mss() as sct:
            scr = make_screenshot(sct)
        self.screenshot_wh = scr.width, scr.height
        self.screen_wh = tuple(pg.size())

    def click(self, ploc: typing.Tuple[int, int], leftbutton: bool):
        sloc = (int(ploc[0] * self.screen_wh[0] / self.screenshot_wh[0]),
                int(ploc[1] * self.screen_wh[1] / self.screenshot_wh[1]))
        pg.moveTo(sloc[0], sloc[1])
        button = 'left' if leftbutton else 'right'
        pg.click(button=button)


# pylint: disable=too-few-public-methods
class ActionPlanner:
    def __init__(self, delay_after: float, bdetector: vb.BoardDetector):
        self.mc = MouseClicker()
        self.delay_after = delay_after
        self.bd = bdetector

    def click_mines(self, board, qidx_mine) -> None:
        """
        Uncover cells using ``MouseClicker`` according to the solutions
        ``qidx_mine``.

        :param board: the board
        :param qidx_mine: of form array([[bx0, by0, m0], [bx1, by1, m1], ...])
               where bxi is the x board coordinate, byi the y board coordinate,
               mi 0 if there's no mine under else 1
        """
        raise NotImplementedError


# pylint: disable=too-few-public-methods
class PlainActionPlanner(ActionPlanner):
    def click_mines(self, _board, qidx_mine):
        blocs = qidx_mine[:, 0], qidx_mine[:, 1]
        plocs_x, plocs_y = self.bd.boardloc_as_pixelloc(blocs)
        for px, py, mine_under in zip(plocs_x, plocs_y, qidx_mine[:, 2]):
            self.mc.click((px, py), not bool(mine_under))
        time.sleep(self.delay_after)


# pylint: disable=too-few-public-methods
class NoFlagActionPlanner(ActionPlanner):
    def click_mines(self, _board, qidx_mine):
        blocs = qidx_mine[:, 0], qidx_mine[:, 1]
        plocs_x, plocs_y = self.bd.boardloc_as_pixelloc(blocs)
        for px, py, mine_under in zip(plocs_x, plocs_y, qidx_mine[:, 2]):
            if not mine_under:
                self.mc.click((px, py), True)
        time.sleep(self.delay_after)


class ChordActionPlanner(ActionPlanner):
    """
    Note on "Chord": Clicking on numbered/satisfied square will open
    all its neighbors.
    """
    def click_mines(self, board, qidx_mine):
        # TODO
        raise NotImplementedError
