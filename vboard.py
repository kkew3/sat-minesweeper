"""
``vboard`` recognizes the mine board by screenshot.
"""

import typing
import pdb
import os
from functools import partial

import numpy as np
import cv2
import scipy.spatial.distance
import PIL.Image as Image
import PIL.ImageGrab

IMGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'imgs')

# deprecated
FACE_TO_CID = dict(zip('012345678fmq', range(12)))


def normalize(image):
    """
    Normalize a uint8 image to [-1.0, 1.0].
    """
    return (image.astype(np.float64) - 128) / 128


def tobw(img, threshold=128):
    return ((img.astype(np.int64) >= threshold) * 255).astype(np.uint8)


def loadimg(filename: str, bw=True):
    """
    Load image as 1-bit black & white image from ``./imgs/``.

    :param filename: the image filename
    :param bw: if ``True``, threshold the image to 1-bit black and white
    :return: a uint8 image
    """
    filename = os.path.join(IMGDIR, filename)
    img = np.array(Image.open(filename).convert('L'), dtype=np.int64)
    if bw:
        img = tobw(img)
    return img


def get_rect_midpoint(top_left, shape):
    return np.array([
        top_left[0] + shape[1] // 2,
        top_left[1] + shape[0] // 2,
    ])


class BoardNotFoundError(Exception): pass


class KeyLinesNotFoundError(Exception): pass


class BoardLocator:
    def __init__(self, upper, lower, left, right, height, width):
        self.upper = upper
        self.lower = lower
        self.left = left
        self.right = right
        self.height = height
        self.width = width

    def as_board(self, screenshot):
        return screenshot[self.upper:self.lower, self.left:self.right]

    def as_cells(self, screenshot):
        # the key vertical and horizontal lines
        klw = np.linspace(self.left, self.right, self.width + 1,
                          dtype=np.int64)
        klh = np.linspace(self.upper, self.lower, self.height + 1,
                          dtype=np.int64)

        # cell width and height
        cw = int(np.median(np.diff(klw)))
        ch = int(np.median(np.diff(klh)))
        cells = []
        for yp in klh[:-1]:
            for xp in klw[:-1]:
                cells.append(np.copy(screenshot[yp:yp + ch, xp:xp + cw]))
        cells = np.stack(cells)
        return cells

    def as_pixel_coordinate(self, board_coordinate):
        """
        Convert board cell (matrix) coordinate to pixel (image) coordinate.
        """
        x, y = board_coordinate
        py = int((x + .5) / self.height * self.lower
                 + (1 - (x + .5) / self.height) * self.upper)
        px = int((y + .5) / self.width * self.right
                 + (1 - (y + .5) / self.width) * self.left)
        return px, py


def detect_board(screenshot: np.ndarray) -> BoardLocator:
    """
    Locate the board by finding three of the four corners.

    :param screenshot: the uint8 screenshot containing the board
    :return: a BoardLocator object
    :raise BoardNotFoundError: if the coordinates of the corners are not
           consistent with each other
    """
    bcorners = map(loadimg, [
        'board_ll.png',
        'board_ur.png',
        'board_lr.png',
    ])
    mt = partial(cv2.matchTemplate, screenshot, method=cv2.TM_SQDIFF)
    locs = map(cv2.minMaxLoc, map(mt, bcorners))
    locs = iter(x[2] for x in locs)
    ll, ur, lr = tuple(map(partial(get_rect_midpoint, shape=(66, 66)), locs))
    if ll[1] != lr[1] or ur[0] != lr[0]:
        raise BoardNotFoundError
    board = screenshot[ur[1]:lr[1], ll[0]:lr[0]]

    cross = loadimg('cross.png')
    res = cv2.matchTemplate(board, cross, cv2.TM_CCOEFF)
    dots = (res >= np.percentile(np.ravel(res), q=99.5))
    bh = np.max(np.sum(np.abs(np.diff(dots, axis=0)), axis=0)) // 2 + 1
    bw = np.max(np.sum(np.abs(np.diff(dots, axis=1)), axis=1)) // 2 + 1
    if bh * bw == 0:
        raise KeyLinesNotFoundError

    return BoardLocator(ur[1], lr[1], ll[0], lr[0], bh, bw)


class SmilyDetector:
    def __init__(self, topleft, bottomright):
        self.topleft = topleft
        self.bottomright = bottomright

        faces = list(map(normalize, map(loadimg, [
            'face.png', 'facew.png', 'facel.png'
        ])))
        faces = np.stack(faces)
        self.faces = faces.reshape((faces.shape[0], -1))
        self.labels = 'ongoing', 'win', 'lost'

    def get_game_stage(self, screenshot):
        smily = screenshot[self.topleft[1]:self.bottomright[1],
                self.topleft[0]:self.bottomright[0]]
        smily = normalize(smily)
        smily = smily.reshape((1, -1))
        D = scipy.spatial.distance.cdist(self.faces, smily)
        stage = self.labels[np.argmin(D, axis=0)[0]]
        return stage

    def get_smily_pixel_location(self):
        return (int((self.topleft[0] + self.bottomright[0]) / 2),
                int((self.topleft[1] + self.bottomright[1]) / 2))


def locate_smily(screenshot) -> SmilyDetector:
    face = loadimg('face.png')
    res = cv2.matchTemplate(screenshot, face, cv2.TM_SQDIFF)
    _, _, topleft, _ = cv2.minMaxLoc(res)
    bottomright = (topleft[0] + face.shape[1],
                   topleft[1] + face.shape[0])
    return SmilyDetector(topleft, bottomright)


class MineMonitorDetector:
    def __init__(self, rightmost, leftmost, topmost, bottommost, top, right):
        self.rm = rightmost
        self.lm = leftmost
        self.tm = topmost
        self.bm = bottommost
        self.top = top
        self.right = right
        self.lookuptable = np.array([
            [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        ])
        self.units = np.array([100, 10, 1])

    def __call__(self, screenshot_gray):
        roi = screenshot_gray[self.tm:self.bm, self.lm:self.rm]
        region = (roi[self.top:, :self.right] > 50)
        vert = np.linspace(0, region.shape[1], 7, dtype=np.int64)
        hori = np.linspace(0, region.shape[0], 5, dtype=np.int64)
        vresults = np.split(region[:, vert[1::2]], hori[1::2], axis=0)
        hresults = np.split(region[hori[1::2], :], vert[1:-1], axis=1)
        vresults = np.stack([np.sum(x, axis=0) > 0 for x in vresults], axis=1)
        hresults = np.stack([np.sum(x, axis=1) > 0 for x in hresults])
        hresults = hresults.reshape((3, 4))
        results = np.concatenate((vresults, hresults), axis=1).astype(np.int64)
        digits = np.argmax(np.matmul(results, self.lookuptable), axis=1)
        return np.dot(digits, self.units)


def locate_mine_monitor(bdetector: BoardLocator, sdetector: SmilyDetector,
                        screenshot):
    rightmost = sdetector.topleft[0]
    leftmost = bdetector.left
    topmost = sdetector.topleft[1]
    bottommost = sdetector.bottomright[1]
    roi = screenshot[topmost:bottommost, leftmost:rightmost]
    top = np.nonzero(np.abs(np.diff(roi[:, 0])))[0][0] + 1
    right = np.nonzero(np.abs(np.diff(roi[-1, :])))[0][0] + 1
    return MineMonitorDetector(rightmost, leftmost, topmost, bottommost,
                               top, right)


def make_screenshot(fake=False, bw=True) -> np.ndarray:
    if fake:
        scr = Image.open('imgs/desktop-bw.png')
    else:
        scr = PIL.ImageGrab.grab()
    scr = np.array(scr.convert('L'))
    if bw:
        scr = tobw(scr)
    return scr


class CellDetector:
    def __init__(self):
        self.labels = '012345678fmmmq'
        self.cids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11]
        templates = np.stack(tuple(map(normalize, map(loadimg, (
            '0.png', '1.png', '2.png', '3.png',
            '4.png', '5.png', '6.png', '7.png',
            '8.png', 'flag.png',
            'm1.png', 'm2.png', 'm3.png',
            'q.png',
        )))))
        self.templates = templates.reshape((templates.shape[0], -1))

    def __call__(self, querycells, as_label=False):
        querycells = normalize(querycells)
        querycells = querycells.reshape((querycells.shape[0], -1))
        D = scipy.spatial.distance.cdist(self.templates, querycells)
        predictions = np.argmin(D, axis=0)
        labels = self.labels if as_label else self.cids
        predictions = [labels[x] for x in predictions]
        if not as_label:
            predictions = np.array(predictions)
        return predictions


def cellid_as_pixelloc(bdetector: BoardLocator, cellid):
    bloc = np.unravel_index(cellid, (bdetector.height, bdetector.width))
    ploc = bdetector.as_pixel_coordinate(bloc)
    return ploc


if __name__ == '__main__':
    import argparse


    def make_parser():
        parser = argparse.ArgumentParser(
            description='Find board and extract cells')
        parser.add_argument('-S', '--screenshot', nargs='?',
                            help='if specified, use the given image; '
                                 'otherwise, take a screenshot')
        parser.add_argument('-E', '--empty-board', metavar='IMAGE',
                            required=True,
                            dest='empty_board', type=os.path.normpath,
                            help='the screenshot of empty board')
        parser.add_argument('-o', '--board-tofile', metavar='FILE',
                            dest='board_tofile', type=os.path.normpath,
                            help='if specified, the board image will be '
                                 'saved to FILE')
        parser.add_argument('-d', '--cells-todir', metavar='DIR',
                            dest='cells_todir', type=os.path.normpath,
                            help='if specified, the cell images will be '
                                 'populated to DIR, named by numeric order')
        parser.add_argument('-B', '--recognized-board-tofile', metavar='FILE',
                            dest='recognized_board_tofile',
                            type=os.path.normpath,
                            help='if specified, the recognized board CSV '
                                 'will be saved to FILE')
        return parser


    def main():
        args = make_parser().parse_args()
        if not args.screenshot:
            scr = make_screenshot()
        else:
            scr = np.array(Image.open(args.screenshot).convert('L'))
            scr = tobw(scr)
        em = np.array(Image.open(args.empty_board).convert('L'))
        em = tobw(em)
        cdetector = CellDetector()
        assert em.shape == scr.shape
        bdetector = detect_board(em)
        if args.cells_todir:
            cells = bdetector.as_cells(scr)
            n_cells = cells.shape[0]
            dwidth = int(np.ceil(np.log10(n_cells)))
            for j, c in enumerate(cells):
                name = str(j).rjust(dwidth, '0')
                tofile = os.path.join(args.cells_todir, name + '.png')
                Image.fromarray(c).save(tofile)
        elif args.board_tofile:
            board = bdetector.as_board(scr)
            Image.fromarray(board).save(args.board_tofile)
        elif args.recognized_board_tofile:
            cells = bdetector.as_cells(scr)
            board = cdetector(cells)
            board = np.array(board)
            board = board.reshape((bdetector.height, bdetector.width))
            np.savetxt(args.recognized_board_tofile, board,
                       fmt='%d', delimiter=',')


    main()
