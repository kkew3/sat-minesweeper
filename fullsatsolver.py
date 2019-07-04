import pdb
import logging
import itertools
import typing
import sys
import io
import shutil

import numpy as np
from pysat.card import CardEnc
from pysat.solvers import Solver as SATSolver

from solverutils import CID
import solverutils as sutils


Clauses = typing.List[typing.List[int]]


def _l(*args):
    return logging.getLogger('.'.join((__name__,) + args))


class NCKProblemEncoder:
    def __init__(self, n_vars_total: int) -> None:
        self.s1 = n_vars_total + 1
        self.logger = _l(NCKProblemEncoder.__name__)

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
        for x, y in zip(*np.where(board == CID['q'])):
            surr = boxof(board, (x, y))
            if np.sum((surr >= 1) & (surr <= 8)) > 0:
                v = int(vartable[x, y])
                qvars_to_use.append(v)
    else:
        for x, y in zip(*np.where(board == CID['q'])):
            qvars_to_use.append(int(vartable[x, y]))
    qvar2vid = dict((v, i+1) for i, v in enumerate(qvars_to_use))
    pe = NCKProblemEncoder(len(qvar2vid))

    for x, y in zip(*np.where((board <= 8) & (board >= 1))):
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
    all_vars = sorted(set(map(abs, itertools.chain.from_iterable(clauses))))
    qvars = list(qvar2vid)
    logger.debug('Involved vars=%s, final clauses=%s', qvars, clauses)
    return qvars, clauses


class TooManySolutionsError(Exception):
    def __init__(self, solutions):
        self.solutions = solutions


class NoSolutionError(Exception):
    pass


def attempt_full_solve(clauses, solver='minisat22', max_solutions=3000):
    with SATSolver(name=solver, bootstrap_with=clauses) as s:
        solutions = list(itertools.islice(s.enum_models(), max_solutions + 1))
    if not solutions:
        raise NoSolutionError
    if len(solutions) == max_solutions + 1:
        raise TooManySolutionsError(solutions)
    return np.array(solutions, dtype=np.int64)


def analyze_solutions(solutions, nv):
    solutions = np.sign(solutions[:, :nv])
    confidence = np.abs(np.sum(solutions, axis=0)) / solutions.shape[0]
    mines = np.sign(np.sum(solutions, axis=0)) >= 0
    return confidence, mines


def attempt_probing(all_vars, clauses, solutions, solver='minisat22', th=1e-6):
    logger = _l(attempt_probing.__name__)
    solutions = np.array(solutions)
    confidence, mine = analyze_solutions(solutions, len(all_vars))
    quasiconfident = np.where(confidence > 1.0 - th)[0]
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

def solve(board: np.ndarray, mines_remain: int = None):
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
                and mines_remain < 5:
            logger.info('No confident decision. Rerunning inference using '
                        'mines_remain')
            qidx_mine, confidence = solve_board(board, mines_remain)
        if np.max(confidence) > uscore:
            return qidx_mine[np.where(confidence > uscore)]
        return qidx_mine[np.argmax(confidence)][np.newaxis]
    except NoSolutionError:
        logger.warning('NoSolutionError')
        logger.info('Falling back to random guess')
        all_blocs = np.stack(np.where(board == CID['q']), axis=1)
        rand_bloc = all_blocs[np.random.randint(all_blocs.shape[0])]
        rand_mine = 0  # if guess 1 it ends up mistaken but found after 
                       # several steps
        logger.info('Choosing: bloc=%s, mine_under=%s', rand_bloc, rand_mine)
        return np.concatenate((rand_bloc, [rand_mine]))[np.newaxis]
    finally:
        logger.info('Inference done')


def _main():
    args = sutils.make_parser().parse_args()
    if args.board_csv:
        infile = open(args.board_csv)
    else:
        infile = sys.stdin
    with io.StringIO() as sbuf:
        shutil.copyfileobj(infile, sbuf)
        sbuf.seek(0)
        firstline = sbuf.readline()
        if firstline.startswith('#mines '):
            mines_remain = int(firstline[len('#mines '):].rstrip())
        else:
            sbuf.seek(0)
            mines_remain = None
        board = np.loadtxt(sbuf, delimiter=',', dtype=np.int64)
    if args.board_csv:
        infile.close()
    try:
        qidx_mine = solve(board, mines_remain)
    except NoSolutionError:
        print('NoSolutionError', file=sys.stderr)
        sys.exit(1)
    else:
        np.savetxt(sys.stdout, qidx_mine, delimiter=',', fmt='%d')


if __name__ == '__main__':
    _main()
