import argparse
import time
import logging
import logging.config
import itertools

import numpy as np

import vboard as vb
import fullsatsolver as satsolver
import actionplanner as planner


class GameWontBeginError(Exception):
    pass


def make_parser():
    parser = argparse.ArgumentParser(description='Minesweeper agent')
    parser.add_argument('-D', dest='delay_before', type=int, default=10,
                        help='seconds to wait before each round; default to '
                             '%(default)s seconds')
    parser.add_argument('-n', dest='rounds', type=int,
                        help='number of rounds to play; default to forever '
                             'until interrupted')
    return parser


def main():
    args = make_parser().parse_args()
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger()

    scr_gray = vb.make_screenshot(bw=False)
    scr = vb.tobw(scr_gray)

    bd = vb.detect_board(scr)
    cd = vb.CellDetector()
    sd = vb.locate_smily(scr)
    md = vb.locate_mine_monitor(bd, sd, scr)
    pl = planner.PlainActionPlanner(0.0, bd, sd)

    mine_remains = md(scr_gray)

    round_total = 0
    round_win = 0

    logger.info('Process begun')
    try:
        for _ in itertools.count(args.rounds):
            logger.info('Waiting %d seconds before current round',
                        args.delay_before)
            time.sleep(args.delay_before)

            scr_gray = vb.make_screenshot(bw=False)
            scr = vb.tobw(scr_gray)
            stage = sd.get_game_stage(scr)
            mine_remains = md(scr_gray)

            if stage != 'ongoing':
                for _ in range(2):
                    stage = pl.click_smily_and_check_stage()
                if stage != 'ongoing':
                    raise GameWontBeginError
            try:
                step = 0
                while stage == 'ongoing':
                    cells = bd.as_cells(scr)
                    board = np.array(cd(cells)).reshape((bd.height, bd.width))
                    logger.debug('Detected board: %s', board.tolist())
                    solutions = satsolver.solve(board, mine_remains)
                    pl.click_mines(solutions)
                    step += 1

                    scr_gray = vb.make_screenshot(bw=False)
                    scr = vb.tobw(scr_gray)
                    stage = sd.get_game_stage(scr)
                    mine_remains = md(scr_gray)
            finally:
                logger.info('Stage: %s', stage)
                round_total += 1
                if stage == 'win':
                    round_win += 1
    except KeyboardInterrupt:
        pass
    except (vb.BoardNotFoundError, vb.KeyLinesNotFoundError,
            GameWontBeginError, satsolver.NoSolutionError):
        logger.exception('')
    except Exception:
        logger.exception('Unexpected exception')
    finally:
        logger.info('Process ended')
        pass


if __name__ == '__main__':
    main()
