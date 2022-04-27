import argparse
import time
import logging
import logging.config

import numpy as np
import PIL.Image as Image
import mss

import vboard as vb
import fullsatsolver as solver
import actionplanner as planner
import solverutils as sutils


class GameWontBeginError(Exception):
    pass


def make_parser():
    parser = argparse.ArgumentParser(description='Minesweeper agent')
    parser.add_argument('-D', dest='delay_before', type=int, default=10,
                        help='seconds to wait before each round; default to '
                             '%(default)s seconds')
    return parser


class BoardFlagModifier:
    def __init__(self):
        self.solutions = {}

    def __call__(self, prev_qidx_mine, board):
        if prev_qidx_mine is not None:
            board = np.copy(board)
            for x, y, m in prev_qidx_mine:
                self.solutions[x, y] = m
            for (x, y), m in self.solutions.items():
                if m:
                    board[x, y] = sutils.CID['f']
        return board


def identify_stage(board):
    if np.sum(board == sutils.CID['m']) > 0:
        return 'lost'
    if np.sum(board == sutils.CID['q']) == 0:
        return 'win'
    return 'ongoing'


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
        pl = planner.NoFlagActionPlanner(0.0, bd)

        logger.info('Process begun')
        try:
            logger.info('Waiting %d seconds before current round',
                        args.delay_before)
            time.sleep(args.delay_before)

            scr = np.array(make_screenshot(sct).convert('L'))
            board, mine_remains = bd.recognize_board_and_mr(scr)
            stage = identify_stage(board)

            if stage != 'ongoing':
                raise GameWontBeginError('game hasn\'t begun yet')
            try:
                step = 0
                bfm = BoardFlagModifier()
                solutions = None
                while stage == 'ongoing':
                    board = bfm(solutions, board)
                    logger.debug('Detected board: %s', board.tolist())
                    solutions = solver.solve(board, mine_remains)
                    pl.click_mines(board, solutions)
                    step += 1

                    scr = np.array(make_screenshot(sct).convert('L'))
                    board, mine_remains = bd.recognize_board_and_mr(scr)
                    stage = identify_stage(board)
            finally:
                logger.info('Stage: %s', stage)
        except KeyboardInterrupt:
            pass
        except (vb.BoardNotFoundError, GameWontBeginError,
                solver.NoSolutionError):
            logger.exception('')
        except Exception:
            logger.exception('Unexpected exception')
        finally:
            logger.info('Process ended')


if __name__ == '__main__':
    main()
