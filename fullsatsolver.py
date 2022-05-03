import logging
import itertools
import typing
import sys

import numpy as np
from pysat.card import CardEnc
from pysat.solvers import Solver as SATSolver

from solverutils import CID
import solverutils as sutils

Clauses = typing.List[typing.List[int]]


def _l(*args):
    return logging.getLogger('.'.join((__name__,) + args))


# pylint: disable=too-few-public-methods
class NCKProblemEncoder:
    def __init__(self, n_vars_total: int):
        self.top_id = n_vars_total

    def __call__(self, vars_, k):
        # these asserts can be safely optimized away if there's no bug upstream
        assert all(x > 0 for x in vars_), \
               'vars_ must be positive but got ' + str(vars_)
        assert max(vars_) <= self.top_id, \
               'max var exceeds top_id {}>{}'.format(max(vars_), self.top_id)
        cnf = CardEnc.equals(vars_, k, top_id=self.top_id)
        # why to use max here is that cnf might not involve auxiliary variable,
        # causing a small cnf.nv (smaller than self.top_id)
        self.top_id = max(self.top_id, cnf.nv)
        C = cnf.clauses
        return C


def encode_board(board: np.ndarray, mine_remains: int = None) \
        -> typing.Tuple[typing.List[int], Clauses]:
    logger = _l(encode_board.__name__)
    vartable = np.arange(1, board.size + 1).reshape(board.shape)
    clauses = set()

    qvars_to_use = []
    if mine_remains is None:
        for x, y in zip(*np.nonzero(board == CID['q'])):
            surr = sutils.boxof(board, (x, y))
            if np.any((surr >= 1) & (surr <= 8)):
                v = int(vartable[x, y])
                qvars_to_use.append(v)
    else:
        for x, y in zip(*np.nonzero(board == CID['q'])):
            qvars_to_use.append(int(vartable[x, y]))
    # qvars_to_use must be in ascending order
    qvar2vid = {v: i for i, v in enumerate(qvars_to_use, start=1)}
    pe = NCKProblemEncoder(len(qvars_to_use))

    for x, y in zip(*np.nonzero((board <= 8) & (board >= 1))):
        surr = sutils.boxof(board, (x, y))
        vsurr = sutils.boxof(vartable, (x, y))
        if np.any(surr == CID['q']):
            vars_ = sorted(vsurr[surr == CID['q']].tolist())
            vars__ = [qvar2vid[x] for x in vars_]
            logger.debug('Translated vars from %s to %s', vars_, vars__)
            k = board[x, y] - np.sum((surr == CID['m']) | (surr == CID['f']))
            C = pe(vars__, k)
            logger.debug('Encoded k=%d vars=%s dcell=%s as clauses C=%s', k,
                         vars__, (x, y), C)
            clauses.update(map(tuple, C))
    if mine_remains is not None:
        vars_ = [qvar2vid[k] for k in qvars_to_use]
        clauses.update(map(tuple, pe(vars_, mine_remains)))
    clauses = list(map(list, clauses))
    logger.debug('Involved vars=%s, final clauses=%s', qvars_to_use, clauses)
    return qvars_to_use, clauses


def attempt_full_solve(clauses, solver='minisat22', max_solutions=10000):
    logger = _l(attempt_full_solve.__name__)
    with SATSolver(name=solver, bootstrap_with=clauses) as s:
        solutions = list(itertools.islice(s.enum_models(), max_solutions + 1))
    # - `[]` occurs when clauses is nonempty and there's no solution
    # - `[[]]` occurs when clauses is empty (for example when inferred mines
    #   are tightly surround the uncovered number cells, and the board cannot
    #   be encoded)
    if solutions in ([], [[]]):
        raise sutils.NoSolutionError
    if len(solutions) == max_solutions + 1:
        logger.warning('TooManySolutionsError. '
                       'There\'s nothing to do about it')
    else:
        logger.debug('Yielded %d solutions', len(solutions))
    return np.array(solutions, dtype=np.int64)


def analyze_solutions(solutions, nv):
    solutions = np.sign(solutions[:, :nv])
    confidence = np.abs(np.sum(solutions, axis=0)) / solutions.shape[0]
    # if confidence == 0, presume there's no mine so that we can proceed.
    # use deterministic strategy here -- don't guess with weight, since the
    # latter is less optimal.
    mines = np.sign(np.sum(solutions, axis=0)) > 0
    return confidence, mines


def solve_board(board: np.ndarray, mines_remain: int = None):
    logger = _l(solve_board.__name__)
    qvars, clauses = encode_board(board, mines_remain)
    solutions = attempt_full_solve(clauses)
    confidence, mine = analyze_solutions(solutions, len(qvars))
    logger.debug('Full solve solutions=%s, confidence=%s', mine.tolist(),
                 confidence.tolist())
    qidx = np.array(qvars) - 1
    qidx = np.stack(np.unravel_index(qidx, board.shape), axis=1)
    logger.debug('Involved blocs=%s', qidx.tolist())
    qidx_mine = np.concatenate((qidx, mine[:, np.newaxis]), axis=1)
    return qidx_mine, confidence


def guess(board: np.ndarray, all_blocs: np.ndarray, guess_edge_weight: float):
    """
    Random guess among all_blocs favoring edges by guess_edge_weight extent.

    :param board: the board
    :param all_blocs: of shape (m, 2) such that the ith row is the board
           coordinate of an empty cell
    :param guess_edge_weight: should be no less than 1.0
    :return: the chosen board coordinate of shape (2,)
    """
    on_edge = ((all_blocs[:, 0] == 0)
               | (all_blocs[:, 0] == board.shape[0] - 1)
               | (all_blocs[:, 1] == 0)
               | (all_blocs[:, 1] == board.shape[1] - 1))
    weights = np.where(on_edge, guess_edge_weight, 1.0)
    weights = weights / np.sum(weights)
    rand_bloc = all_blocs[np.random.choice(
        np.arange(all_blocs.shape[0]), p=weights)]
    return rand_bloc


def solve(board: np.ndarray,
          mines_remain: int = None,
          consider_mines_th: int = 5,
          guess_edge_weight: float = 2.0,
          _first_bloc=None):
    """
    Solve the board.

    :param board: the board
    :param mines_remain: if not None, should be the number of mines not
           uncovered
    :param consider_mines_th: when `mines_remain` is not None and is no
           greater than this number, `mines_remain` is taken into
           consideration
    :param guess_edge_weight: when in the middle of a game and when a random
           guess is required, assign non-edge cells weight 1.0 and assign
           edge cells this weight to perform a weighted guess. Generally
           this weight should be larger than 1.0. This strategy comes from
           the Guessing section of http://www.minesweeper.info/wiki/Strategy
    """
    logger = _l(solve.__name__)
    if np.all(board == CID['q']):
        logger.info('Performing first step random guess')
        if _first_bloc:
            randbloc = _first_bloc
        else:
            randbloc = np.unravel_index(
                np.random.randint(board.size), board.shape)
        logger.info('Choosing bloc=%s', randbloc)
        return np.concatenate((randbloc, [0]))[np.newaxis]
    if np.all(board != CID['q']):
        logger.warning('No uncovered cells found. Why isn\'t the game ended?')
        return np.array([])

    try:
        logger.info('Performing SAT inference')
        qidx_mine, confidence = solve_board(board)
        uscore = 1.0 - 1e-6
        if np.max(confidence) <= uscore and mines_remain is not None \
                and mines_remain <= consider_mines_th:
            logger.info('No confident decision. Rerunning inference using '
                        'mines_remain')
            qidx_mine, confidence = solve_board(board, mines_remain)
        if np.max(confidence) > uscore:
            return qidx_mine[np.nonzero(confidence > uscore)]
        if not np.allclose(np.max(confidence), 0.0):
            return qidx_mine[np.argmax(confidence)][np.newaxis]
        # confidence == [0.0, 0.0, ...], mines should be [False, False, ...]
        assert not np.any(qidx_mine[:, 2]), qidx_mine
        logger.info('Confidences are all zero; failing back to random guess')
        rand_bloc = guess(board, qidx_mine[:, :2], guess_edge_weight)
        logger.info('Choosing: bloc=%s, mine_under=0', rand_bloc)
        return np.concatenate((rand_bloc, [0]))[np.newaxis]
    except sutils.NoSolutionError:
        logger.warning('NoSolutionError')
        logger.info('Falling back to random guess')
        # guess edges with more probability
        all_blocs = np.stack(np.nonzero(board == CID['q']), axis=1)
        rand_bloc = guess(board, all_blocs, guess_edge_weight)
        # if guess 1 it ends up mistaken but found after several steps
        rand_mine = 0
        logger.info('Choosing: bloc=%s, mine_under=%s', rand_bloc, rand_mine)
        return np.concatenate((rand_bloc, [rand_mine]))[np.newaxis]
    finally:
        logger.info('Inference done')


def _main():
    args = sutils.make_parser().parse_args()
    try:
        try:
            board, mines_remain, first_bloc = sutils.read_board(args.board_csv)
        except sutils.EmptyCsvError:
            print('EmptyCsvError', file=sys.stderr)
            return 4
        qidx_mine = solve(board, mines_remain, _first_bloc=first_bloc)
        np.savetxt(sys.stdout, qidx_mine, delimiter=',', fmt='%d')
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        sys.stderr.close()
    return 0


if __name__ == '__main__':
    sys.exit(_main())
