import argparse
import time
import logging
import logging.config
import importlib
import inspect

import numpy as np
from PIL import Image
import mss
import pyautogui as pg

import vboard as vb
import actionplanner as planner
import solverutils as sutils

# world champion's clicking speed, approximately
pg.PAUSE = 0.07


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
    parser.add_argument(
        '-S',
        '--solver',
        choices=['fullsatsolver', 'mcdfssolver', 'mcsatsolver'],
        default='fullsatsolver',
        help='the solver to use; default to %(default)s')
    parser.add_argument(
        'additional_kwargs',
        metavar='KEY=VALUE',
        nargs='*',
        help=('additional keyword arguments to be passed to the solver; see '
              'the solver\'s `solve` function\'s keyword arguments for '
              'detail'))
    return parser


def parse_additional_kwargs(solver: str, kvpairs: list):
    logger = logging.getLogger(__name__ + '.parse_additional_kwargs')
    solve_function = importlib.import_module(solver).solve
    accepted_kwargs = {
        p.name: p.annotation
        for p in inspect.signature(solve_function).parameters.values()
        if p.default is not inspect.Parameter.empty
        and not p.name.startswith('_')
    }
    if any(a is inspect.Parameter.empty for a in accepted_kwargs.values()):
        raise ValueError(
            'the solve function of `{}` misses type annotation '
            'in one of its public keyword arguments'.format(solver))
    parsed_kwargs = {}
    for kvpair in kvpairs:
        k, _, v = kvpair.partition('=')
        if k not in accepted_kwargs:
            logger.warning('Additional kwarg %s is not accepted; skipped', k)
            continue
        parsed_kwargs.update({k: accepted_kwargs[k](v)})
    return parsed_kwargs


def get_mon_resolution(sct, mon_id):
    mon = sct.monitors[mon_id]
    return mon['width'], mon['height']


def main():
    args = make_parser().parse_args()
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger()
    solver = importlib.import_module(args.solver)
    solver_kwargs = parse_additional_kwargs(args.solver,
                                            args.additional_kwargs)
    logger.info('Additional solver kwarge: %s', solver_kwargs)

    with mss.mss() as sct:
        scrs = [(i, get_mon_resolution(sct, i), vb.make_screenshot(sct, i))
                for i in range(1, len(sct.monitors))]
        # leave `enable_mr_detect` False intentionally
        bd = vb.BoardDetector.new(scrs)
        mc = planner.ChrfLBMouseClicker(sct.monitors[bd.mon_id], bd.dpr, bd,
                                        sct)
        pl = planner.GreedyChordActionPlanner(0.0, mc, sct)
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
                    solutions = solver.solve(board, mine_remains,
                                             **solver_kwargs)
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
