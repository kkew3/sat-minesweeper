"""
Recognizes the mine board from screenshot.
"""

import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
import cv2
from PIL import Image
from solverutils import CID

IMGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'imgs')

# related to board cells localization
DOTS_TOL = 200  # the max allowed template matching difference
# related to open cell recognition
OPEN_THR = 153  # the brightness between digit (122) and background (188)
# related to remaining mines digit recognition
MR_LOOKUPTABLE = np.array([
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
]) * 2 - 1
# related to remaining mines digit recognition
MR_UNITS = np.array([100, 10, 1])


def normalize(image):
    """
    Normalize a uint8 image to [-1.0, 1.0].
    """
    return (image.astype(np.float64) - 128) / 128


def tobw(img, threshold):
    return ((img.astype(np.int64) >= threshold) * 255).astype(np.uint8)


def loadimg(filename: str):
    """
    Load image as grayscale from ``IMGDIR``.

    :param filename: the image filename
    :return: a uint8 image
    """
    filename = os.path.join(IMGDIR, filename)
    img = np.asarray(Image.open(filename).convert('L'))
    return img


def get_rect_midpoint(top_left, shape):
    return np.array([
        top_left[0] + shape[1] // 2,
        top_left[1] + shape[0] // 2,
    ])


def make_screenshot(sct, monitor=None, region=None):
    """
    Make uint8 grayscale screenshot of specified region on specified monitor.

    :param sct: the ``mss.mss()`` instance
    :param monitor: ``None`` for the first monitor, positive integer for the
           monitor of that id, and dict for that monitor
    :type monitor: Union[None, int, dict]
    :param region: ``None`` for the entire region, and dict for the specified
           region plus the offset imposed by the specified monitor
    :return: numpy array of the grayscale screenshot
    """
    if isinstance(monitor, int):
        monitor = sct.monitors[monitor]
    elif not monitor:
        monitor = sct.monitors[1]
    if region:
        region['top'] += monitor['top']
        region['left'] += monitor['left']
        img = sct.grab(region)
    else:
        img = sct.grab(monitor)
    img = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
    return np.asarray(img.convert('L'))


class BoardNotFoundError(Exception):
    """
    Raised when the board cells cannot be segmented out correctly.
    """
    pass


class BoardDetector:
    """
    Attributes (note: the x-y coordinate complies to image convention):

        - ``upper``: the smallest y coordinate of the board (readonly)
        - ``lower``: the largest y coordinate of the board (readonly)
        - ``left``: the smallest x coordinate of the board (readonly)
        - ``right``: the largest x coordinate of the baord (readonly)
        - ``height``: the number of cells along each column (readonly)
        - ``width``: the number of cells along each row (readonly)
        - ``hkls``: horizontal key lines of the cell board
        - ``vkls``: vertical key lines of the cell board

    Below attributes may be ``None`` if ``enable_mr_detect=False`` when
    ``new``:

        - ``upper_mr``: the smallest y coordinate of the remaining mines label
        - ``lower_mr``: the largest y coordinate of the remaining mines label
        - ``left_mr``: the smallest x coordinate of the remaining mines label
        - ``right_mr``: the largest x coordinate of the remaining mines label
    """
    def __init__(self, mon_id, dpr, hkls, vkls, upper_mr, lower_mr, left_mr,
                 right_mr):
        """
        This method shouldn't be called explicitly.
        """
        # the monitor id
        self.mon_id = mon_id
        # the device pixel ratio (x, y)
        self.dpr = dpr

        # the cell board key lines
        self.hkls = hkls
        self.vkls = vkls

        # the remaining mines label location
        self.upper_mr = upper_mr
        self.lower_mr = lower_mr
        self.left_mr = left_mr
        self.right_mr = right_mr

        # precomputed board region and remaining mines region
        self.board_region = {
            'top': self.upper,  # self.upper is a property
            'left': self.left,  # same
            'width': self.right - self.left,  # same
            'height': self.lower - self.upper,  # same
        }
        if self.upper_mr is not None:
            self.mr_region = {
                'top': self.upper_mr // self.dpr[1],
                'left': self.left_mr // self.dpr[0],
                'width': (self.right_mr - self.left_mr) // self.dpr[0],
                'height': (self.lower_mr - self.upper_mr) // self.dpr[1],
            }
        else:
            self.mr_region = None

        # precomputed offset hkls and vkls, i.e. the key lines with respect
        # to the upper left corner of the board region
        self.offset_hkls = self.hkls - self.hkls[0]
        self.offset_vkls = self.vkls - self.vkls[0]

        # preload various cells
        loaded_imgs = [
            tobw(loadimg('open{}.gif'.format(i)), OPEN_THR)
            for i in range(0, 9)
        ]
        loaded_imgs.extend(
            map(loadimg, [
                'bombflagged.gif', 'bombdeath.gif', 'bombmisflagged.gif',
                'bombrevealed.gif', 'blank.gif'
            ]))
        self._face_templates = np.stack(loaded_imgs).astype(np.float64)
        self._face_templates = self._face_templates / 255 * 2 - 1
        self._face_templates = self._face_templates.reshape(
            self._face_templates.shape[0], -1)
        self._face_templates_cids = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            CID['f'],
            CID['m'],
            CID['m'],
            CID['m'],
            CID['q'],
        ]

    @property
    def upper(self):
        return self.hkls[0] // self.dpr[1]

    @property
    def lower(self):
        return self.hkls[-1] // self.dpr[1]

    @property
    def left(self):
        return self.vkls[0] // self.dpr[0]

    @property
    def right(self):
        return self.vkls[-1] // self.dpr[0]

    @property
    def height(self):
        """Board height, not pixel height"""
        return self.hkls.size - 1

    @property
    def width(self):
        """Board width, not pixel width"""
        return self.vkls.size - 1

    def __str__(self):
        return ('{0.__class__.__name__}('
                'mon_id={0.mon_id}, '
                'dpr={0.dpr}, '
                'hkls={0.hkls}, '
                'vkls={0.vkls}, '
                'upper_mr={0.upper_mr}, '
                'lower_mr={0.lower_mr}, '
                'left_mr={0.left_mr}, '
                'right_mr={0.right_mr})'.format(self))

    def __repr__(self):
        return ('{0.__class__.__name__}('
                'mon_id={0.mon_id} '
                'dpr={0.dpr}, '
                'hkls={0.hkls!r}, '
                'vkls={0.vkls!r}, '
                'upper_mr={0.upper_mr!r}, '
                'lower_mr={0.lower_mr!r}, '
                'left_mr={0.left_mr!r}, '
                'right_mr={0.right_mr!r})'.format(self))

    @classmethod
    def new(cls, mon_screenshots, enable_mr_detect=False):
        """
        Try every pair of (monitor id, monitor resolution, screenshot) until
        one returns an instance of ``BoardDetector``.

        :param mon_screenshots: list of tuples of (monitor id, monitor
               resolution (width, height), the uint8 grayscale screenshot
               possibly containing an empty board)
        :param enable_mr_detect: if ``True``, enable mines remaining detection
        :return: a ``BoardDetector`` object
        :raise BoardNotFoundError: if until the last monitor ``BoardDetector``
               is not instantiated successfully
        """
        total_num = len(mon_screenshots)
        for i, (mon_id, mon_res, screenshot) in enumerate(mon_screenshots, 1):
            try:
                return cls._new(mon_id, mon_res, screenshot, enable_mr_detect)
            except BoardNotFoundError:
                if i == total_num:
                    raise

    @classmethod
    def _new(cls, mon_id: int, mon_res, screenshot: np.ndarray,
             enable_mr_detect):
        """
        Returns a new instance of ``BoardDetector`` from ``screenshot``.

        :param mon_id: the monitor id
        :param mon_res: the monitor resolution (width, height)
        :param screenshot: the uint8 grayscale screenshot containing an empty
               board
        :param enable_mr_detect: if ``True``, enable mines remaining detection
        :return: a ``BoardDetector`` object
        :raise BoardNotFoundError:
        """
        # COMPUTE DEVICE PIXEL RATIO
        dpr_x = screenshot.shape[1] // mon_res[0]
        dpr_y = screenshot.shape[0] // mon_res[1]

        # LOCALIZE CELL BOARD
        crosstmpl = loadimg('b_crs.png')
        mmr = cv2.matchTemplate(screenshot, crosstmpl,
                                cv2.TM_SQDIFF) <= DOTS_TOL
        dots = np.stack(np.nonzero(mmr), axis=1)
        if dots.size == 0:
            raise BoardNotFoundError('no board cross is found')
        u0, cnt0 = np.unique(dots[:, 0], return_counts=True)
        u1, cnt1 = np.unique(dots[:, 1], return_counts=True)
        # remove outliers
        cnt0_e, cnt0_c = np.unique(cnt0, return_counts=True)
        cnt0_mode = cnt0_e[np.argmax(cnt0_c)]
        cnt1_e, cnt1_c = np.unique(cnt1, return_counts=True)
        cnt1_mode = cnt1_e[np.argmax(cnt1_c)]
        to_delete = [
            np.where(dots[:, 0] == x)[0] for x in u0[cnt0 < cnt0_mode]
        ] + [np.where(dots[:, 1] == x)[0] for x in u1[cnt1 < cnt1_mode]]
        if to_delete:
            dots = np.delete(
                dots, np.unique(np.concatenate(to_delete)), axis=0)

        ch_ = np.unique(np.diff(np.unique(dots[:, 0])))  # cell intervals y
        cw_ = np.unique(np.diff(np.unique(dots[:, 1])))  # cell intervals x
        # allow one unique dot interval or two successive dot intervals due
        # to rounding error
        if not ((ch_.size == 1 or
                 (ch_.size == 2 and abs(ch_[0] - ch_[1]) == 1)) and
                (cw_.size == 1 or
                 (cw_.size == 2 and abs(cw_[0] - cw_[1]) == 1))):
            raise BoardNotFoundError('board crosses are not localized '
                                     'correctly')
        # the horizontal (arranged along matrix axis=0) key lines
        hkls = np.unique(dots[:, 0])
        hkls = np.concatenate((
            [hkls[0] - (hkls[1] - hkls[0])],
            hkls,
            [hkls[-1] + (hkls[-1] - hkls[-2])],
        )) + 1
        # the vertical (arranged along matrix axis=1) key lines
        vkls = np.unique(dots[:, 1])
        vkls = np.concatenate((
            [vkls[0] - (vkls[1] - vkls[0])],
            vkls,
            [vkls[-1] + (vkls[-1] - vkls[-2])],
        )) + 1
        if not enable_mr_detect:
            return cls(mon_id, (dpr_x, dpr_y), hkls, vkls, None, None, None,
                       None)

        left = vkls[0]
        right = vkls[-1]

        # LOCALIZE MINE REMAINING LABEL
        mrlltmpl = loadimg('mr_ll.png')
        mrlrtmpl = loadimg('mr_lr.png')
        mrultmpl = loadimg('mr_ul.png')
        MR_TOL = 50
        mrllloc = np.stack(
            np.nonzero(
                cv2.matchTemplate(screenshot, mrlltmpl, cv2.TM_SQDIFF) <=
                MR_TOL),
            axis=1)
        mrlrloc = np.stack(
            np.nonzero(
                cv2.matchTemplate(screenshot, mrlrtmpl, cv2.TM_SQDIFF) <=
                MR_TOL),
            axis=1)
        mrulloc = np.stack(
            np.nonzero(
                cv2.matchTemplate(screenshot, mrultmpl, cv2.TM_SQDIFF) <=
                MR_TOL),
            axis=1)
        mrlrloc = np.delete(
            mrlrloc, np.where(mrlrloc[:, 1] >= np.mean((left, right))), axis=0)
        mrulloc = np.delete(
            mrulloc, np.where(mrulloc[:, 1] >= np.mean((left, right))), axis=0)
        if mrllloc.size > 0 and abs(mrllloc[0, 1] - left + 1) <= 1:
            mrllloc[0, 1] = left - 1
        if mrulloc.size > 0 and abs(mrulloc[0, 1] - left + 1) <= 1:
            mrulloc[0, 1] = left - 1
        if (any(x.shape[0] != 1 for x in (mrllloc, mrlrloc, mrulloc))
                or mrllloc[0, 1] != left - 1 or mrllloc[0, 0] != mrlrloc[0, 0]
                or mrulloc[0, 1] != left - 1):
            raise BoardNotFoundError('remaining mines label is not localized '
                                     'correctly')
        lower_mr, left_mr = mrllloc[0] + 1
        upper_mr = mrulloc[0, 0] + 1
        right_mr = mrlrloc[0, 1] + 1

        return cls(mon_id, (dpr_x, dpr_y), hkls, vkls, upper_mr, lower_mr,
                   left_mr, right_mr)

    def recognize_board_and_mr(self, sct):
        boardimg, mrimg = self.localize_board_and_mr(sct)
        cellimgs = self.get_cells_from_board(boardimg)
        cells = self.recognize_cells(cellimgs)
        if self.upper_mr is None:
            mr = None
        else:
            mr = self.recognize_mr_digits(mrimg)
        return cells, mr, boardimg

    @staticmethod
    def recognize_mr_digits(roi_gray):
        region = roi_gray > 50
        vert = np.linspace(0, region.shape[1], 7, dtype=np.int64)
        hori = np.linspace(0, region.shape[0], 5, dtype=np.int64)
        vresults = np.split(region[:, vert[1::2]], hori[1::2], axis=0)
        hresults = np.split(region[hori[1::2], :], vert[1:-1], axis=1)
        vresults = np.stack([np.sum(x, axis=0) > 0 for x in vresults], axis=1)
        hresults = np.stack([np.sum(x, axis=1) > 0 for x in hresults])
        hresults = hresults.reshape((3, 4))
        results = np.concatenate((vresults, hresults), axis=1).astype(np.int64)
        digits = np.argmax(np.matmul(results * 2 - 1, MR_LOOKUPTABLE), axis=1)
        return np.dot(digits, MR_UNITS)

    def localize_board_and_mr(self, sct):
        """
        Returns ``(cell_board_image, mine_remaining_image)`` if
        ``enable_mr_detect`` was ``True`` when calling ``new`` to construct
        this ``BoardDetector``; otherwise, returns
        ``(cell_board_image, None)``.
        """
        boardimg = make_screenshot(sct, self.mon_id, self.board_region)
        if self.upper_mr is None:
            return boardimg, None
        mrimg = make_screenshot(sct, self.mon_id, self.mr_region)
        return boardimg, mrimg

    def get_cells_from_board(self, boardimg):
        cells = []
        for i in range(self.offset_hkls.size - 1):
            for j in range(self.offset_vkls.size - 1):
                # yapf: disable
                c = boardimg[self.offset_hkls[i]:self.offset_hkls[i + 1],
                             self.offset_vkls[j]:self.offset_vkls[j + 1]]
                # yapf: enable
                cells.append(np.copy(c))
        cells = np.stack(cells)
        return cells

    def recognize_cells(self, cells):
        cells = np.stack(
            [tobw(cv2.resize(x, (16, 16)), OPEN_THR) for x in cells])
        cells = cells.astype(np.float64) / 255 * 2 - 1
        cells = cells.reshape((cells.shape[0], -1))
        D = cdist(self._face_templates, cells)
        predictions = np.argmin(D, axis=0)
        predictions = [self._face_templates_cids[x] for x in predictions]
        predictions = np.array(predictions).reshape((self.height, self.width))
        return predictions

    def boardloc_as_pixelloc(self, blocs):
        """
        Convert a batch of board locations to a batch of pixel locations. Note
        that in the board coordinate x axis is from the upper left corner to
        the lower left corner and the y axis is from the upper left corner to
        the upper right corner; whereas in the pixel coordinate x axis is from
        the upper left corner to the upper right corner, etc.

        :param blocs: of form (array([...], dtype=int), array([...], dtype=int)
               where the first array is the board x coordinates, and the
               second array the board y coordinates
        :return: pixel coordinates of the same form as ``blocs``
        """
        bx, by = blocs
        py = ((self.hkls[bx] + self.hkls[bx + 1]) / 2).astype(int)
        px = ((self.vkls[by] + self.vkls[by + 1]) / 2).astype(int)
        return px, py

    @staticmethod
    def _cc_dist(query, templates):
        return min(
            abs(x.astype(np.int64) - query.astype(np.int64))
            for x in templates)


# pylint: disable=too-few-public-methods
class StageIdentifier:
    def identify_stage(self, scr, board):
        """
        :param scr: should be an array of shape (H, W), of dtype uint8
        :param board: the recognized board
        """
        min_white_ratio = 1/3  # minimum required ratio of white pixels
        sample_size = 32  # size of center crop
        assert scr.shape[0] > sample_size and scr.shape[1] > sample_size
        splower = (scr.shape[0] - sample_size) // 2
        spleft = (scr.shape[1] - sample_size) // 2
        spl = scr[splower:splower + sample_size, spleft:spleft + sample_size]
        # if the winning message appears, there should be many white pixels
        # within the crop region
        if np.sum(spl > 250) / spl.size > min_white_ratio:
            return 'win'
        if np.any(board == CID['m']):
            return 'lost'
        return 'ongoing'


def _main():
    parser = argparse.ArgumentParser(
        description='Recognize board from screenshot.')
    parser.add_argument(
        '-R',
        dest='empty_board',
        type=os.path.normpath,
        help='recognize from screenshot given EMPTY_BOARD in '
        'scene if specified; otherwise, localize board '
        'and mine remaining label from screenshot')
    parser.add_argument(
        '-D',
        dest='empty_board_monitor',
        type=int,
        default=1,
        help='the monitor id of the empty_board')
    parser.add_argument(
        '-b',
        type=os.path.normpath,
        dest='board_tofile',
        metavar='FILE',
        help='if specified, the board image will be saved to '
        'FILE')
    parser.add_argument(
        '-m',
        type=os.path.normpath,
        dest='mr_tofile',
        metavar='FILE',
        help='if specified, the mine remaining image will be '
        'saved to FILE')
    parser.add_argument(
        '-B',
        action='store_true',
        dest='boardcsv',
        help='if specified, the board CSV will be printed to screen')
    parser.add_argument(
        '-C',
        type=os.path.normpath,
        dest='cellnpy_tofile',
        metavar='FILE',
        help='if specified, the cell images are zipped in an npy FILE')
    parser.add_argument(
        '-M',
        action='store_true',
        dest='mrnum',
        help='if specified, the mine remaining number will '
        'be printed on screen')
    args = parser.parse_args()

    with mss.mss() as sct:

        def get_mon_resolution(_mon_id):
            _mon = sct.monitors[_mon_id]
            return _mon['width'], _mon['height']

        if not args.empty_board:
            empty_board = [(i, get_mon_resolution(i), make_screenshot(sct, i))
                           for i in range(1, len(sct.monitors))]
        else:
            empty_board = [
                (
                    args.empty_board_monitor,
                    get_mon_resolution(args.empty_board_monitor),
                    np.asarray(Image.open(args.empty_board).convert('L')),
                ),
            ]
        bd = BoardDetector.new(empty_board, args.mr_tofile or args.mrnum)
        boardimg, mrimg = bd.localize_board_and_mr(sct)
    if args.board_tofile:
        Image.fromarray(boardimg).save(args.board_tofile)
    if args.mr_tofile:
        Image.fromarray(mrimg).save(args.mr_tofile)
    if args.empty_board is not None and args.boardcsv:
        np.savetxt(
            sys.stdout,
            bd.recognize_cells(bd.get_cells_from_board(boardimg)),
            fmt='%d',
            delimiter=',')
    if args.mrnum:
        print(bd.recognize_mr_digits(mrimg))
    if args.cellnpy_tofile:
        np.save(args.cellnpy_tofile, bd.get_cells_from_board(boardimg))
    print(bd)


if __name__ == '__main__':
    import argparse
    import mss
    _main()
