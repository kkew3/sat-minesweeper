import typing
import time

import numpy as np
import pyautogui as pg
import PIL.Image as Image
import mss

import vboard as vb


def make_screenshot(sct):
    img = sct.grab(sct.monitors[1])
    img = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
    return img


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


class ActionPlanner:
    def __init__(self, delay_after: float, bdetector: vb.BoardLocator):
        self.mc = MouseClicker()
        self.delay_after = delay_after
        self.bd = bdetector

    def click_mines(self, board, qidx_mine):
        raise NotImplementedError


class PlainActionPlanner(ActionPlanner):
    def __init__(self, delay_after: float, bdetector: vb.BoardLocator):
        super().__init__(delay_after, bdetector)

    def click_mines(self, board, qidx_mine):
        blocs = np.ravel_multi_index(qidx_mine[:, :2].T,
                                     (self.bd.height, self.bd.width))
        for bloc, mine_under in zip(blocs, qidx_mine[:, 2]):
            ploc = vb.cellid_as_pixelloc(self.bd, bloc)
            self.mc.click(ploc, not bool(mine_under))
        time.sleep(self.delay_after)


class NoFlagActionPlanner(ActionPlanner):
    def __init__(self, delay_after: float, bdetector: vb.BoardLocator):
        super().__init__(delay_after, bdetector)

    def click_mines(self, board, qidx_mine):
        blocs = np.ravel_multi_index(qidx_mine[:, :2].T,
                                     (self.bd.height, self.bd.width))
        for bloc, mine_under in zip(blocs, qidx_mine[:, 2]):
            if not mine_under:
                ploc = vb.cellid_as_pixelloc(self.bd, bloc)
                self.mc.click(ploc, True)
        time.sleep(self.delay_after)


class AreaOpenActionPlanner(ActionPlanner):
    """
    Note on "AreaOpen": Clicking on numbered/satisfied square will open
    all its neighbors.
    """
    def __init__(self, delay_after: float, bdetector: vb.BoardLocator):
        super().__init__(delay_after, bdetector)

    def click_mines(self, board, qidx_mine):
        # TODO
        raise NotImplementedError
