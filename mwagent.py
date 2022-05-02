import argparse
import time
import logging
import logging.config

import numpy as np
from PIL import Image
import cv2
import mss
import pyautogui as pg

import vboard as vb
import actionplanner as planner
import solverutils as sutils

# Select your favorite solver here, by commenting and uncommenting
import fullsatsolver as solver
#import mcdfssolver as solver

# world champion's clicking speed, approximately
pg.PAUSE = 0.05


class GameWontBeginError(Exception):
    def __init__(self):
        super().__init__('game hasn\'t begun yet')


def make_parser():
    parser = argparse.ArgumentParser(description='Minesweeper agent')
    parser.add_argument(
        '-D',
        dest='delay_before',
        type=int,
        default=10,
        help='seconds to wait before each round; default to '
        '%(default)s seconds')
    parser.add_argument(
        '-m',
        dest='num_mines',
        type=int,
        metavar='N',
        help='total number of mines')
    return parser


# Deprecated. This function does not adapt to the new interface of
# minesweeper.org
#def identify_stage(scr, board):
#    if np.sum(board == sutils.CID['m']) > 0:
#        return 'lost'
#    if np.sum(board == sutils.CID['q']) == 0:
#        return 'win'
#    return 'ongoing'


def make_screenshot(sct):
    img = sct.grab(sct.monitors[1])
    img = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
    return img


def main():
    args = make_parser().parse_args()
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger()

    with mss.mss() as sct:
        scr = np.array(make_screenshot(sct).convert('L'))
        bd = vb.BoardDetector.new(scr)
        pl = planner.GreedyChordActionPlanner(0.0, bd, sct)
        si = vb.StageIdentifier()

        logger.info('Process begun')
        try:
            logger.info('Waiting %d seconds before current round',
                        args.delay_before)
            time.sleep(args.delay_before)

            board, _, boardimg = bd.recognize_board_and_mr(sct)
            stage = si.identify_stage(boardimg, board)

            if stage != 'ongoing':
                raise GameWontBeginError
            tic = time.time()
            try:
                step = 0
                bfm = planner.ChordBoardFlagModifier()
                solutions = None
                while stage == 'ongoing':
                    board = bfm(solutions, board)
                    logger.debug('Detected board: %s', board.tolist())
                    if args.num_mines:
                        mine_remains = args.num_mines - np.sum(
                            board == sutils.CID['f'])
                        logger.info('# Mine remains: %d', mine_remains)
                    else:
                        mine_remains = None
                    solutions = solver.solve(board, mine_remains)
                    board = bfm.rewind_board()
                    pl.click_mines(board, solutions)
                    step += 1

                    board, _, boardimg = bd.recognize_board_and_mr(sct)
                    stage = si.identify_stage(boardimg, board)
            finally:
                toc = time.time()
                logger.info('Stage: %s', stage)
                logger.info('Time used: %f seconds', toc - tic)
        except KeyboardInterrupt:
            pass
        except (vb.BoardNotFoundError, GameWontBeginError):
            logger.exception('')
        except Exception:
            logger.exception('Unexpected exception')
        finally:
            logger.info('Process ended')


if __name__ == '__main__':
    main()
