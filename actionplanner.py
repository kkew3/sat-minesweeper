import typing
import time

import numpy as np
import pyautogui as pg

import vboard as vb


class MouseClicker:
    def __init__(self):
        scr = vb.make_screenshot(bw=False)
        self.screenshot_wh = scr.shape[::-1]
        self.screen_wh = tuple(pg.size())

    def click(self, ploc: typing.Tuple[int, int], leftbutton: bool):
        sloc = (int(ploc[0] * self.screen_wh[0] / self.screenshot_wh[0]),
                int(ploc[1] * self.screen_wh[1] / self.screenshot_wh[1]))
        pg.moveTo(sloc[0], sloc[1])
        button = 'left' if leftbutton else 'right'
        pg.click(button=button)


class ActionPlanner:
    def __init__(self, delay_after: float, bdetector: vb.BoardLocator,
                 sdetector: vb.SmilyDetector):
        self.mc = MouseClicker()
        self.delay_after = delay_after
        self.bd = bdetector
        self.sd = sdetector

    def click_smily_and_check_stage(self):
        sloc = self.sd.get_smily_pixel_location()
        self.mc.click(sloc, True)
        time.sleep(0.5)
        scr = vb.make_screenshot()
        stage = self.sd.get_game_stage(scr)
        return stage

    def click_mines(self, board, qidx_mine):
        raise NotImplementedError


class PlainActionPlanner(ActionPlanner):
    def __init__(self, delay_after: float, bdetector: vb.BoardLocator,
                 sdetector: vb.SmilyDetector):
        super().__init__(delay_after, bdetector, sdetector)

    def click_mines(self, board, qidx_mine):
        blocs = np.ravel_multi_index(qidx_mine[:,:2].T,
                                     (self.bd.height, self.bd.width))
        for bloc, mine_under in zip(blocs, qidx_mine[:,2]):
            ploc = vb.cellid_as_pixelloc(self.bd, bloc)
            self.mc.click(ploc, not bool(mine_under))
        time.sleep(self.delay_after)


class NoFlagActionPlanner(ActionPlanner):
    def __init__(self, delay_after: float, bdetector: vb.BoardLocator,
                 sdetector: vb.SmilyDetector):
        super().__init__(delay_after, bdetector, sdetector)

    def click_mines(self, board, qidx_mine):
        blocs = np.ravel_multi_index(qidx_mine[:,:2].T,
                                     (self.bd.height, self.bd.width))
        for bloc, mine_under in zip(blocs, qidx_mine[:,2]):
            if not mine_under:
                ploc = vb.cellid_as_pixelloc(self.bd, bloc)
                self.mc.click(ploc, True)
        time.sleep(self.delay_after)


class AreaOpenActionPlanner(ActionPlanner):
    """
    Note on "AreaOpen": Clicking on numbered/satisfied square will open
    all its neighbors.
    """
    def __init__(self, delay_after: float, bdetector: vb.BoardLocator,
                 sdetector: vb.SmilyDetector):
        super().__init__(delay_after, bdetector, sdetector)

    def click_mines(self, board, qidx_mine):
        # TODO
        raise NotImplementedError
