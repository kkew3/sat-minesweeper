import typing
import time
import sys

import pyautogui as pg


# pylint: disable=too-few-public-methods
class MouseClicker:
    def __init__(self, bdetector):
        """
        :type bdetector: simulated.vboard.BoardDetector
        """
        self.bd = bdetector

    def click(self, ploc: typing.Tuple[int, int], leftbutton: bool):
        if leftbutton:
            self.bd.left_click_cell(ploc)
        else:
            self.bd.flag_cell(ploc)
        time.sleep(pg.PAUSE)
        if sys.platform == 'darwin':
            time.sleep(pg.DARWIN_CATCH_UP_TIME)
