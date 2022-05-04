import argparse
import time
import logging
import logging.config
import importlib

import numpy as np
import pyautogui as pg

import virtual.vboard as vb
import actionplanner as planner
from virtual.actionplanner import MouseClicker
import solverutils as sutils
import mwagent

# world champion's clicking speed, approximately
pg.PAUSE = 0.05


def make_parser():
    parser = argparse.ArgumentParser(description='Virtual Minesweeper agent')
    parser.add_argument(
        '-D',
        dest='delay_before',
        type=int,
        default=10,
        help='seconds to wait before each round; default to '
        '%(default)s seconds')
    parser.add_argument(
        '-S',
        '--solver',
        choices=['fullsatsolver', 'mcdfssolver', 'mcsatsolver'],
        default='fullsatsolver',
        help='the solver to use; default to %(default)s')
    parser.add_argument(
        'key_board',
        metavar='KEY_BOARD_CSV',
        help=('the board to play on; should be of the same format as '
              'specified in solverutils'))
    parser.add_argument(
        'additional_kwargs',
        metavar='KEY=VALUE',
        nargs='*',
        help=('additional keyword arguments to be passed to the solver; see '
              'the solver\'s `solve` function\'s keyword arguments for '
              'detail'))
    return parser


def main():
    args = make_parser().parse_args()
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger()
    solver = importlib.import_module(args.solver)
    solver_kwargs = mwagent.parse_additional_kwargs(args.solver,
                                                    args.additional_kwargs)
    logger.info('Additional solver kwarge: %s', solver_kwargs)

    key_board, total_mines, first_bloc = sutils.read_board(args.key_board)
    if not first_bloc:
        raise ValueError('first_bloc not specified in KEY_BOARD_CSV')

    bd = vb.BoardDetector.new(key_board)
    pl = planner.GreedyChordActionPlanner(0.0, bd, None, _mc=MouseClicker(bd))
    si = vb.StageIdentifier(key_board)

    logger.info('Process begun')
    try:
        logger.info('Waiting %d seconds before current round',
                    args.delay_before)
        time.sleep(args.delay_before)

        board, _, boardimg = bd.recognize_board_and_mr(None)
        stage = si.identify_stage(boardimg, board)

        if stage != 'ongoing':
            raise mwagent.GameWontBeginError
        tic = time.time()
        try:
            step = 0
            bfm = planner.ChordBoardFlagModifier()
            solutions = None
            while stage == 'ongoing':
                board = bfm(solutions, board)
                logger.debug('Detected board: %s', board.tolist())
                if total_mines:
                    mine_remains = total_mines - np.sum(
                        board == sutils.CID['f'])
                    logger.info('# Mine remains: %d', mine_remains)
                else:
                    mine_remains = None
                solutions = solver.solve(
                    board,
                    mine_remains,
                    _first_bloc=first_bloc,
                    **solver_kwargs)
                board = bfm.rewind_board()
                pl.click_mines(board, solutions)
                step += 1

                board, _, boardimg = bd.recognize_board_and_mr(None)
                stage = si.identify_stage(boardimg, board)
        finally:
            toc = time.time()
            logger.info('Stage: %s', stage)
            logger.info('Time used: %f seconds', toc - tic)
    except KeyboardInterrupt:
        pass
    except mwagent.GameWontBeginError:
        logger.exception('')
    except Exception:
        logger.exception('Unexpected exception')
    finally:
        logger.info('Process ended')


if __name__ == '__main__':
    main()
