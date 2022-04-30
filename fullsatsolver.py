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
    def __init__(self, n_vars_total: int) -> None:
        self.s1 = n_vars_total + 1
        self.logger = _l(NCKProblemEncoder.__name__)

    # TODO there may be some redundant code during encoding because I didn't
    #      read much into `pysat.card.CardEnc.equals`'s documentation
    def __call__(self, k, vars_):
        assert max(map(abs, vars_)) < self.s1, 'max vars exceeded nvars_total'
        assert len(vars_) == len(set(vars_)), 'vars_ element must be unique'
        assert all(x > 0 for x in vars_), 'vars_ must be greater than 0'
        n = len(vars_)
        cnf = CardEnc.equals(range(1, n + 1), k)
        C = cnf.clauses
        # ns: number of auxiliary varibales in C
        ns = len(set(map(abs, itertools.chain.from_iterable(C)))) - n
        assert ns >= 0, (ns, n, C, k, vars_)
        sdiff = self.s1 - (n + 1)
        assert sdiff >= 0, (self.s1, n, sdiff, k, vars_)
        if sdiff:
            C = [[(x + sdiff) if x > n else x for x in r] for r in C]
            C = [[(x - sdiff) if x < -n else x for x in r] for r in C]
        tr = dict(zip(range(1, n + 1), vars_))
        tr.update(dict(zip(range(-1, -n - 2, -1), [-x for x in vars_])))
        C = [[tr.get(x, x) for x in r] for r in C]
        self.s1 += ns
        self.logger.debug('Updated self.s1 from %d to %d',
                          self.s1 - ns, self.s1)
        C = [[int(x) for x in r] for r in C]
        return C


def boxof(array, center, radius=1):
    return array[max(0, center[0]-radius):center[0]+radius+1,
                 max(0, center[1]-radius):center[1]+radius+1]


def encode_board(board: np.ndarray, mine_remains: int = None) \
        -> typing.Tuple[typing.List[int], Clauses]:
    logger = _l(encode_board.__name__)
    vartable = np.arange(1, board.size + 1).reshape(board.shape)
    clauses = set()

    qvars_to_use = []
    if mine_remains is None:
        for x, y in zip(*np.nonzero(board == CID['q'])):
            surr = boxof(board, (x, y))
            if np.any((surr >= 1) & (surr <= 8)):
                v = int(vartable[x, y])
                qvars_to_use.append(v)
    else:
        for x, y in zip(*np.nonzero(board == CID['q'])):
            qvars_to_use.append(int(vartable[x, y]))
    qvar2vid = dict((v, i+1) for i, v in enumerate(qvars_to_use))
    pe = NCKProblemEncoder(len(qvar2vid))

    for x, y in zip(*np.nonzero((board <= 8) & (board >= 1))):
        surr = boxof(board, (x, y))
        vsurr = boxof(vartable, (x, y))
        if np.sum(surr == CID['q']) > 0:
            vars_ = sorted(vsurr[surr == CID['q']].tolist())
            vars__ = [qvar2vid[x] for x in vars_]
            logger.debug('Translated vars from %s to %s', vars_, vars__)
            vars_ = vars__
            k = board[x, y] - np.sum((surr == CID['m']) | (surr == CID['f']))
            C = pe(k, vars_)
            logger.debug('Encoded k=%d vars=%s dcell=%s as clauses C=%s',
                         k, vars_, (x, y), C)
            clauses.update(map(tuple, C))
    if mine_remains is not None:
        vars_ = [qvar2vid[k] for k in sorted(qvar2vid)]
        clauses.update(map(tuple, pe(mine_remains, vars_)))
    clauses = list(map(list, clauses))
    qvars = list(qvar2vid)
    logger.debug('Involved vars=%s, final clauses=%s', qvars, clauses)
    return qvars, clauses


class TooManySolutionsError(Exception):
    def __init__(self, solutions):
        super().__init__()
        self.solutions = solutions


class NoSolutionError(Exception):
    pass


def attempt_full_solve(clauses, solver='minisat22', max_solutions=3000):
    with SATSolver(name=solver, bootstrap_with=clauses) as s:
        solutions = list(itertools.islice(s.enum_models(), max_solutions + 1))
    # - `[]` occurs when clauses is nonempty and there's no solution
    # - `[[]]` occurs when clauses is empty (for example when inferred mines
    #   are tightly surround the uncovered number cells, and the board cannot
    #   be encoded)
    if solutions in ([], [[]]):
        raise NoSolutionError
    if len(solutions) == max_solutions + 1:
        raise TooManySolutionsError(solutions)
    return np.array(solutions, dtype=np.int64)


def analyze_solutions(solutions, nv):
    solutions = np.sign(solutions[:, :nv])
    confidence = np.abs(np.sum(solutions, axis=0)) / solutions.shape[0]
    # if confidence == 0, presume there's no mine so that we can proceed
    mines = np.sign(np.sum(solutions, axis=0)) > 0
    return confidence, mines


def attempt_probing(all_vars, clauses, solutions, solver='minisat22', th=1e-6):
    logger = _l(attempt_probing.__name__)
    solutions = np.array(solutions)
    confidence, mine = analyze_solutions(solutions, len(all_vars))
    quasiconfident = np.nonzero(confidence > 1.0 - th)[0]
    with SATSolver(name=solver, bootstrap_with=clauses) as s:
        for i in map(int, quasiconfident):
            neg = s.solve(assumptions=[-(i + 1)])
            pos = s.solve(assumptions=[i + 1])
            if neg and not pos:
                logger.debug('+%d (a.k.a +%d) excluded', i + 1, all_vars[i])
                mine[i] = False
                confidence[i] = 1.0
            elif pos and not neg:
                logger.debug('-%d (a.k.a. -%d) excluded', i + 1, all_vars[i])
                mine[i] = True
                confidence[i] = 1.0
            else:
                logger.debug('%d (a.k.a. %d) undecided', i + 1, all_vars[i])
                confidence[i] = 0.0
    return confidence, mine


def solve_board(board: np.ndarray, mines_remain: int = None):
    logger = _l(solve_board.__name__)
    qvars, clauses = encode_board(board, mines_remain)
    try:
        solutions = attempt_full_solve(clauses)
    except TooManySolutionsError as e:
        logger.debug('Handling TooManySolutionError')
        confidence, mine = attempt_probing(qvars, clauses, e.solutions,
                                           th=0.2)
        logger.debug('Assumption solve solutions=%s, confidence=%s',
                     mine.tolist(), confidence.tolist())
    else:
        confidence, mine = analyze_solutions(solutions, len(qvars))
        logger.debug('Full solve solutions=%s, confidence=%s', mine.tolist(),
                     confidence.tolist())
    qidx = np.array(qvars) - 1
    qidx = np.stack(np.unravel_index(qidx, board.shape), axis=1)
    qidx_mine = np.concatenate((qidx, mine[:, np.newaxis]), axis=1)
    return qidx_mine, confidence

def solve(board: np.ndarray, mines_remain: int = None,
          consider_mines_th: int = 5, guess_edge_weight: float = 2.0):
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
        # guess edges with more probability
        all_blocs = qidx_mine[:, :2]
        on_edge = ((all_blocs[:, 0] == 0)
                   | (all_blocs[:, 0] == board.shape[0] - 1)
                   | (all_blocs[:, 1] == 0)
                   | (all_blocs[:, 1] == board.shape[1] - 1))
        weights = np.where(on_edge, guess_edge_weight, 1.0)
        weights = weights / np.sum(weights)
        rand_bloc = all_blocs[np.random.choice(np.arange(all_blocs.shape[0]),
                                               p=weights)]
        logger.info('Choosing: bloc=%s, mine_under=0', rand_bloc)
        return np.concatenate((rand_bloc, [0]))[np.newaxis]
    except NoSolutionError:
        logger.warning('NoSolutionError')
        logger.info('Falling back to random guess')
        # guess edges with more probability
        all_blocs = np.stack(np.nonzero(board == CID['q']), axis=1)
        on_edge = ((all_blocs[:, 0] == 0)
                   | (all_blocs[:, 0] == board.shape[0] - 1)
                   | (all_blocs[:, 1] == 0)
                   | (all_blocs[:, 1] == board.shape[1] - 1))
        weights = np.where(on_edge, guess_edge_weight, 1.0)
        weights = weights / np.sum(weights)
        rand_bloc = all_blocs[np.random.choice(np.arange(all_blocs.shape[0]),
                                               p=weights)]
        rand_mine = 0  # if guess 1 it ends up mistaken but found after
                       # several steps
        logger.info('Choosing: bloc=%s, mine_under=%s', rand_bloc, rand_mine)
        return np.concatenate((rand_bloc, [rand_mine]))[np.newaxis]
    finally:
        logger.info('Inference done')


def _main():
    args = sutils.make_parser().parse_args()
    try:
        try:
            board, mines_remain = sutils.read_board(args.board_csv)
        except sutils.EmptyCsvError:
            print('EmptyCsvError', file=sys.stderr)
            sys.exit(4)
        try:
            qidx_mine = solve(board, mines_remain)
        except NoSolutionError:
            print('NoSolutionError', file=sys.stderr)
            sys.exit(8)
        else:
            np.savetxt(sys.stdout, qidx_mine, delimiter=',', fmt='%d')
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        sys.stderr.close()


if __name__ == '__main__':
    _main()
