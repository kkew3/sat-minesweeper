import pdb
import typing
import itertools
import os
import logging
import tempfile
import contextlib
import subprocess
import json
import string

import numpy as np

from vboard import FACE_TO_CID

CID_M = FACE_TO_CID['m']
CID_F = FACE_TO_CID['f']
CID_Q = FACE_TO_CID['q']

MAX_TRANSLATED_SYMBOL_POW26 = 2

mid_symbols = list(map(''.join, itertools.product(
    *[string.ascii_lowercase
      for _ in range(MAX_TRANSLATED_SYMBOL_POW26)])))


def translate_cnf(cnf, src, dest, mid=None):
    if len(src) != len(dest):
        raise ValueError(f'lengths of symbol sets must be equal: '
                         f'len(src)={len(src)}, len(dest)={len(dest)}')
    if mid is not None and len(src) != len(mid):
        raise ValueError(f'lengths of symbol sets must be equal: '
                         f'len(src)={len(src)}, len(mid)={len(mid)}')
    if mid is None:
        if len(src) > len(mid_symbols):
            raise ValueError(f'too large the symbol set '
                             f'({len(src)}>{len(mid_symbols)})')
        mid = mid_symbols[:len(src)]

    src = sorted(src, key=len, reverse=True)
    logger = logging.getLogger('.'.join((__name__, 'translate_cnf')))
    logger.debug('Symbol set translating ongoing: %s -> %s -> %s',
                 src, mid, dest)

    sbuf = json.dumps(cnf)
    for s, m in zip(src, mid):
        sbuf = sbuf.replace(s, m)
    for m, d in zip(mid, dest):
        sbuf = sbuf.replace(m, d)
    cnf = json.loads(sbuf)
    return cnf, (dest, src)


class CNFTemplateLib:
    def __init__(self):
        with open('data/sattable.json') as infile:
            self.cnfs = json.load(infile)
        self.index = {}
        self.index = {
            tuple(label): i for i, (label, _) in enumerate(self.cnfs)
        }
        logger = logging.getLogger('.'.join((__name__, 'CNFTemplateLib')))
        logger.debug('Loaded from data/sattable.json')

    def get(self, vars_: typing.Sequence[int], k: int):
        """
        :param vars_: a list of variables
        :param k: number of mines among the variables
        :raise KeyError: if the combination of vars_ and k is too complicated
               to solve
        :return: the CNF w.r.t. vars_
        """
        n = len(vars_)
        assert n <= 10, vars_
        cnf = self.cnfs[self.index[(n, k)]][1]
        srcch = list(map(str, range(1, n+1)))
        tarch = list(map(str, vars_))
        cnf, reverse_tr = translate_cnf(cnf, srcch, tarch)
        return cnf, reverse_tr


def first_step(predicted_board):
    """
    :return: the cell unravelled coordinate of the first step
    """
    return np.random.randint(predicted_board.size), False


class CNF:
    def __init__(self, vars_, cclauses=None):
        self.cclauses = cclauses
        self.vars_ = vars_
        if cclauses is not None:
            self.cano_cclauses, self.reverse_tr = translate_cnf(
                self.cclauses, list(map(str, self.vars_)),
                list(map(str, range(1, len(self.vars_) + 1))))

    def as_dimac_cnffile(self, filename=None):
        """
        Return a (temporary) file handle where the DIMAC CNF file has been
        written to and to be read from. The file needs to be closed by the
        invoker function.

        :param filename: if given, the returned file handle will be a
               non-temporary file named ``filename``
        """
        if self.cclauses is None:
            raise RuntimeError('cclauses not specified; nothing to write')

        if not filename:
            outfile = tempfile.NamedTemporaryFile(mode='r+', delete=False)
        else:
            filename = os.path.normpath(filename)
            outfile = open(filename, 'w+')

        outfile.write(f'p cnf {len(self.vars_)} {len(self.cclauses)}\n')
        for cl in self.cano_cclauses:
            outfile.write(' '.join(map(str, cl)))
            outfile.write(' 0\n')
        outfile.seek(0)
        return outfile


class GameLostError(Exception): pass


def make_cnf(templates: CNFTemplateLib, predicted_board) -> CNF:
    logger = logging.getLogger('.'.join((__name__, 'make_cnf')))
    numcells = (predicted_board >= 1) & (predicted_board <= 8)
    n_minecells = np.sum(predicted_board == CID_M)
    if n_minecells > 0:
        raise GameLostError

    cclauses = set()
    vars_ = set()
    vartable = np.arange(1, predicted_board.size + 1) \
            .reshape(predicted_board.shape)
    for x, y in zip(*np.where(numcells)):
        logger.debug('Processing (x,y)=(%d,%d)', x, y)
        neighbors = predicted_board[max(0,x-1):x+2,max(0,y-1):y+2]
        logger.debug('Neighbors: %s', neighbors)
        qneighbors = np.where(neighbors == CID_Q)
        if qneighbors[0].shape[0]:
            dclause_vars = [vartable[max(0,x-1):x+2,max(0,y-1):y+2][sx,sy]
                            for sx, sy in zip(*qneighbors)]
            k = predicted_board[x, y] - len(np.where(neighbors == CID_F)[0])
            logger.debug('DNF variables: %s; # of mines: %d', dclause_vars, k)
            try:
                cc, _ = templates.get(dclause_vars, k)
                logger.debug('Fetched from templatelib clauses: %s', cc)
            except KeyError:
                logger.warning('Failed to fetch CNF template from lib')
                continue
            cclauses.update(map(tuple, cc))
            logger.debug('Total clauses: %s', cclauses)
            vars_.update(dclause_vars)
            logger.debug('Total symbols: %s', vars_)
    cclauses = list(map(list, cclauses))
    cnf = CNF(vars_, cclauses)
    logger.debug('Final clauses: %s', cclauses)
    logger.debug('Final canonical clauses: %s', cnf.cano_cclauses)
    logger.debug('Final symbols: %s', vars_)
    return cnf

def run_picosat(cnf):
    logger = logging.getLogger('.'.join((__name__, 'run_picosat')))
    solutions = []
    with contextlib.closing(cnf.as_dimac_cnffile()) as tmpfile:
        args = ('bin/picosat', '--all', tmpfile.name)
        logger.debug('Running picosat')
        with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
            sbuf = []
            for line in proc.stdout:
                line = line.decode('ascii').rstrip()
                if line.startswith('v '):
                    logger.debug('Reading solution line: %s', line)
                    if line.endswith(' 0'):
                        sbuf.append(line[2:-2])
                        sbuf = ' '.join(sbuf)
                        logger.debug('Final line buffer: %s', sbuf)
                        solutions.append(list(map(int, sbuf.split())))
                        sbuf = []
                    else:
                        sbuf.append(line[2:])
        logger.debug('Solver returns')
    logger.debug('Solutions: %s', solutions)
    return solutions


class NoSolutionError(Exception): pass


def interprete_solutions(solutions):
    logger = logging.getLogger('.'.join((__name__, 'interprete_solutions')))
    assert len(set(map(len, solutions))) == 1
    solutions = np.array([sorted(l, key=abs) for l in solutions])
    logger.debug('Solutions: %s', solutions)
    assert np.all(np.repeat(np.arange(1, solutions.shape[1] + 1)[np.newaxis],
                            solutions.shape[0], axis=0)
                  - np.abs(solutions) == 0)
    if solutions.size:
        signs = np.sign(solutions)
        logger.debug('Solution signs: %s', signs)
        colsums = np.abs(np.sum(signs, axis=0))
        sorted_indices = np.argsort(colsums)
        actionpairs = []
        for index in sorted_indices[::-1]:
            idx = index + 1
            deterministic = (colsums[index] == signs.shape[0])
            if deterministic:
                value = (signs[0, index] > 0)
                actionpairs.append((idx, value))
            elif not actionpairs:
                value = (np.median(signs[:, index]) > 0)
                actionpairs.append((idx, value))
                break
            else:
                deterministic = True
                break

        logger.debug('Selected index-value pairs: %s', actionpairs)
        return deterministic, actionpairs
    else:
        raise NoSolutionError


def solution_as_cellid(cnf, idx, has_mine):
    cellid = int(dict(zip(*cnf.reverse_tr))[str(idx)]) - 1
    return cellid, has_mine
