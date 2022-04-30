# Deprecated
"""
Divide the whole SAT problem into many independent subproblems and perform
partial assignments.
"""
import multiprocessing
import pdb
from functools import partial
import itertools
import json
import logging

import numpy as np
from pysat.solvers import Minisat22

from vboard import FACE_TO_CID

CID_Q = FACE_TO_CID['q']
CID_F = FACE_TO_CID['f']
CID_M = FACE_TO_CID['m']


def first_step(board):
    """
    :return: the cell unravelled coordinate of the first step
    """
    return np.random.randint(board.size), False


def random_guess(board):
    qcells = list(zip(*np.where(board == CID_Q)))
    cellcoor = np.random.randint(len(qcells))
    cellid = np.ravel_multi_index(cellcoor, board.shape)
    guess_value = int(np.random.randint(2))
    actionpair = [(cellid, bool(guess_value))]
    return actionpair


class UnexpectedGameLostError(Exception):
    pass


class NoSolutionError(Exception):
    pass


def build_index(board):
    """
    :param board: predicted board
    :return: (central_variable => (# mines, query_variables),
              central_variable => neighbor_variables)
    """
    index = {}
    vartable = np.arange(1, board.size + 1).reshape(board.shape)
    for x, y in zip(*np.where((board >= 1) & (board <= 8))):
        surr = board[max(0, x - 1):x + 2, max(0, y - 1):y + 2]
        msurr = np.where(surr == CID_M)
        if msurr[0].shape[0]:
            raise UnexpectedGameLostError
        qsurr = np.where(surr == CID_Q)
        if qsurr[0].shape[0]:
            v = vartable[x, y]
            qvars = [vartable[max(0, x - 1):x + 2, max(0, y - 1):y + 2][sx, sy]
                     for sx, sy in zip(*qsurr)]
            fsurr = np.where(surr == CID_F)
            n_mines = board[x, y]
            rem_mines = n_mines - fsurr[0].shape[0]
            index[v] = (rem_mines, qvars)
    neighbor_index = {}
    for v in index:
        # noinspection PyTupleAssignmentBalance
        x, y = next(zip(*np.where(vartable == v)))
        surr = board[max(0, x - 2):x + 3, max(0, y - 2):y + 3]
        dsurr = tuple(
            vartable[max(0, x - 2):x + 3, max(0, y - 2):y + 3][sx, sy]
            for sy in range(surr.shape[1])
            for sx in range(surr.shape[0])
            if vartable[max(0, x - 2):x + 3, max(0, y - 2):y + 3][sx, sy]
            in (set(index) - {vartable[x, y]}))
        neighbor_index[v] = dsurr
    return index, neighbor_index


def translate_clauses(clauses, tr):
    tr_ = {-k: -v for k, v in tr.items()}
    tr_.update(tr)
    return [[tr_[x] for x in cl] for cl in clauses]


class CNFTemplateLib:
    def __init__(self):
        with open('data/sattable.json') as infile:
            self.cnfs = json.load(infile)
        self.index = {}
        self.index = {tuple(label): i
                      for i, (label, _) in enumerate(self.cnfs)}
        logger = logging.getLogger('.'.join((__name__, 'CNFTemplateLib')))
        logger.debug('Loaded from data/sattable.json')

    def get(self, n: int, k: int):
        """
        :param n: number of variables
        :param k: number of mines among the variables
        :raise KeyError: if the combination of n and k is too complicated
               to solve
        :return: the canonical CNF
        """
        cnf = self.cnfs[self.index[(n, k)]][1]
        return cnf


def solve_neighborhood_cnf(corevars, allvars, concat_clauses,
             SolverClass=Minisat22, max_assignments=1000):
    tr = dict(zip(allvars, range(1, len(allvars) + 1)))
    concat_clauses = translate_clauses(concat_clauses, tr)
    with SolverClass(bootstrap_with=concat_clauses) as s:
        solutions = list(itertools.islice(s.enum_models(), max_assignments))
    if not solutions:
        raise NoSolutionError
    itr = dict(zip(range(1, len(allvars) + 1), allvars))
    solutions = list(map(partial(sorted, key=abs), solutions))
    solutions = translate_clauses(solutions, itr)
    coreidx = list(map(list(map(abs, solutions[0])).index, corevars))
    coresolutions = (np.array(solutions)[:, coreidx] > 0)
    set_sums = np.sum(coresolutions, axis=0)
    determ0 = np.where(set_sums == 0)[0]
    dvars0 = [x for i, x in enumerate(corevars) if i in determ0]
    determ1 = np.where(set_sums == len(solutions))[0]
    dvars1 = [x for i, x in enumerate(corevars) if i in determ1]
    if dvars0 or dvars1:
        return True, (dvars0, dvars1)
    ndvar_idx = np.argmax(np.maximum(set_sums, len(solutions) - set_sums))
    if len(solutions) - set_sums[ndvar_idx] > set_sums[ndvar_idx]:
        ndvars = ([corevars[ndvar_idx]], [])
    else:
        ndvars = ([], [corevars[ndvar_idx]])
    return False, ndvars, abs(set_sums[ndvar_idx] / len(solutions) - 0.5)


def solve(templates: CNFTemplateLib, index, neighbor_index, *,
          pool: multiprocessing.Pool = None):
    singlecnfs = {}
    for v in index:
        k = index[v][0]
        n = len(index[v][1])
        try:
            cnf = templates.get(n, k)
        except KeyError:
            singlecnfs[v] = None
        else:
            tr = dict(zip(range(1, n + 1), index[v][1]))
            singlecnfs[v] = translate_clauses(cnf, tr)
    args = []
    for v in neighbor_index:
        if singlecnfs[v] is not None:
            clauses = set(map(tuple, singlecnfs[v]))
            corevars = set(index[v][1])
            allvars = set()
            allvars.update(corevars)
            for nv in neighbor_index[v]:
                if singlecnfs[nv] is not None:
                    allvars.update(index[nv][1])
                    clauses.update(map(tuple, singlecnfs[nv]))
            clauses = list(map(list, clauses))
            corevars = sorted(corevars)
            allvars = sorted(allvars)
            args.append((corevars, allvars, clauses))
    if pool is None:
        solutions = list(itertools.starmap(solve_neighborhood_cnf, args))
    else:
        solutions = list(pool.starmap(solve_neighborhood_cnf, args, 100))
    if any(x[0] for x in solutions):
        return True, [x[1] for x in solutions[:] if x[0]]
    try:
        return False, [sorted(solutions, key=lambda x: x[2])[-1][1]]
    except IndexError:
        raise NoSolutionError


def interprete_solutions(solutions):
    logger = logging.getLogger('.'.join(
        (__name__, 'interprete_solutions')))
    zeros = []
    ones = []
    for zl, ol in solutions:
        zeros.extend(zl)
        ones.extend(ol)
    zeros = set(zeros)
    ones = set(ones)
    if zeros & ones:
        logger.warning('Conflicting solution exists; removing them')
        zeros_ = zeros - ones
        ones_ = ones - zeros
        zeros, ones = zeros_, ones_
    results = ([(idx-1, False) for idx in zeros]
               + [(idx-1, True) for idx in ones])
    return results
