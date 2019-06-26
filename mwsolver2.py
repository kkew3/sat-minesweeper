import multiprocessing
import time
import logging
import logging.config
import argparse

import numpy as np
import pyautogui as pg

import vboard as vb
import multisatinfer

PAUSE_AFTER_CLICK = 0.3


def positive_int(string: str) -> int:
    try:
        value = int(string)
    except ValueError:
        raise argparse.ArgumentTypeError
    if value <= 0:
        raise argparse.ArgumentTypeError
    return value

def make_parser():
    parser = argparse.ArgumentParser(
        description='Automatic mineweeper solver based on Gaussian reduction')
    parser.add_argument('-D', '--delay', type=positive_int, dest='delay',
                        default=10,
                        help='seconds to delay working loop so that the '
                             'mineweeper interface is in the front; default '
                             'to %(default)s')
    return parser


def configure_logging():
    logging.config.fileConfig('logging.ini')


def pixelcoor_to_screencoor(screenshot_wh, screen_wh, ploc):
    return (int(ploc[0] * screen_wh[0] / screenshot_wh[0]),
            int(ploc[1] * screen_wh[1] / screenshot_wh[1]))


class MouseClicker:
    def __init__(self, screenshot_wh, screen_wh):
        self.screenshot_wh = screenshot_wh
        self.screen_wh = screen_wh
        self.logger = logging.getLogger('.'.join((__name__, 'MouseClicker')))

    def click(self, ploc, mine_under, wait=True):
        sloc = pixelcoor_to_screencoor(self.screenshot_wh, self.screen_wh, ploc)
        pg.moveTo(sloc[0], sloc[1])
        button = 'right' if mine_under else 'left'
        pg.click(button=button)
        if wait:
            time.sleep(PAUSE_AFTER_CLICK)
        self.logger.debug('Clicked %s button at %s (ploc=%s)',
                          button, sloc, ploc)

    @staticmethod
    def wait():
        time.sleep(PAUSE_AFTER_CLICK)


def action(bdetector, mouse_clicker: MouseClicker, actionpair) -> None:
    logger = logging.getLogger('.'.join((__name__, 'action')))
    for bloc_unraveled, mine_under in actionpair:
        ploc = vb.cellid_as_pixelloc(bdetector, bloc_unraveled)
        logger.info('Ready to click cell id %d with %d mine under',
                    bloc_unraveled, mine_under)
        mouse_clicker.click(ploc, mine_under, wait=False)
    mouse_clicker.wait()


class GameWontBeginError(Exception): pass


def main():
    configure_logging()
    logger = logging.getLogger()
    args = make_parser().parse_args()

    logger.info('Process started')

    cdetector = vb.CellDetector()
    cnf_tm = multisatinfer.CNFTemplateLib()
    screenshot_wh = vb.make_screenshot().shape[::-1]
    screen_wh = tuple(pg.size())
    mouse_clicker = MouseClicker(screenshot_wh, screen_wh)
    round_win = 0
    round_total = 0

    scr = vb.make_screenshot()
    bdetector = vb.detect_board(scr)
    sdetector = vb.locate_smily(scr)
    pool = multiprocessing.Pool(4)

    try:
        while True:
            logger.info('Waiting for next round for %d seconds', args.delay)
            time.sleep(args.delay)
            stage = sdetector.get_game_stage(scr)
            logger.info('Stage: %s', stage)
            if stage != 'ongoing':
                logger.info('Trying to click the face')
                smily_loc = sdetector.get_smily_pixel_location()
                mouse_clicker.click(smily_loc, False)
                logger.debug('Clicked face at %s', smily_loc)
                scr = vb.make_screenshot()
                stage = sdetector.get_game_stage(scr)
                if stage != 'ongoing':
                    logger.info('Trying to click the face again')
                    smily_loc = sdetector.get_smily_pixel_location()
                    mouse_clicker.click(smily_loc, False)
                    logger.debug('Clicked face at %s', smily_loc)
                    scr = vb.make_screenshot()
                    stage = sdetector.get_game_stage(scr)
                    if stage != 'ongoing':
                        raise GameWontBeginError
            try:
                step = 0
                while stage == 'ongoing':
                    logger.info('Step %d', step)
                    cells = bdetector.as_cells(scr)
                    board = cdetector(cells)
                    board = np.array(board)
                    board = board.reshape((bdetector.height, bdetector.width))
                    logger.debug('Detected board (shape=%dx%d): %s',
                                 bdetector.height, bdetector.width,
                                 board.tolist())
                    if np.all(board == multisatinfer.CID_Q):
                        logger.info('Performing first step random guess')
                        actionpair = [multisatinfer.first_step(board)]
                        deterministic = False
                    else:
                        logger.info('Performing SAT inference')
                        indexes = multisatinfer.build_index(board)
                        try:
                            deterministic, sol = multisatinfer.solve(
                                cnf_tm, *indexes, pool=pool)
                        except multisatinfer.NoSolutionError:
                            logger.warning('No solution was found; falling '
                                           'back to random guess')
                            actionpair = multisatinfer.random_guess(board)
                            deterministic = False
                        else:
                            actionpair = multisatinfer.interprete_solutions(sol)
                    action(bdetector, mouse_clicker, actionpair)
                    logger.info('Solution was selected deterministically: %s',
                                deterministic)
                    step += 1

                    scr = vb.make_screenshot()
                    stage = sdetector.get_game_stage(scr)
                    logger.info('Stage: %s', stage)
            except multisatinfer.UnexpectedGameLostError:
                logger.warning('Game lost unexpectedly')
                stage = 'lost'
            finally:
                pool.close()
                round_total += 1
                if stage == 'win':
                    round_win += 1
                logger.info('Win rate: %f (%d/%d)', round_win/round_total,
                            round_win, round_total)
    except KeyboardInterrupt:
        logger.info('Interrupted')
    except (vb.BoardNotFoundError,
            vb.KeyLinesNotFoundError,
            GameWontBeginError):
        logger.exception('')
    except Exception:
        logger.exception('Unexpected exception')
    finally:
        logger.info('Process ended')


if __name__ == '__main__':
    main()
