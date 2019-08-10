"""
DFS solver based on Kernighan-Lin partition algorithm to reduce recursion
complexity.
"""

import pdb
import sys
import itertools
import collections
import logging
import functools
import operator
import contextlib

import networkx as nx
import numpy as np

import solverutils as sutils
from solverutils import CID

MAX_VARS = 32
MAX_ITER = 2 ** 24  # takes approximately 1 second

NCKProblem = collections.namedtuple('NCKProblem', 'vars k')


# a placeholder
class NoSolutionError(Exception): pass


def encode_board_as_nckproblems(board, mines_remain):
    vartable = np.arange(board.size).reshape(board.shape)
    problems = []
    for x, y in zip(*np.where((board >= 1) & (board <= 8))):
        box = board[max(0, x - 1):x + 2, max(0, y - 1):y + 2]
        vbox = vartable[max(0, x - 1):x + 2, max(0, y - 1):y + 2]
        if np.sum(box == CID['q']) > 0:
            problems.append(NCKProblem(
                vars=tuple(vbox[sx, sy] for sx, sy in
                           zip(*np.where(box == CID['q']))),
                k=(board[x, y] - np.sum(box == CID['m'])
                   - np.sum(box == CID['f'])),
            ))
            assert problems[-1].k >= 0, problems[-1]
    if mines_remain is not None:
        mproblem = NCKProblem(vars=tuple(vartable[sx, sy] for sx, sy in
                                         zip(*np.where(board == CID['q']))),
                              k=mines_remain)
    else:
        mproblem = None
    return problems, mproblem


def reduce_problem(solutions, problem):
    new_vars = tuple(v for v in problem.vars if v not in solutions)
    new_k = problem.k - sum(solutions[v] for v in problem.vars
                            if v in solutions)
    assert new_k >= 0, (solutions, problem)
    if not new_vars:
        return None
    return NCKProblem(vars=new_vars, k=new_k)


def simple_solve_attempt(problems, mproblem):
    unsolved_problems = []
    solutions = {}
    updated = True
    while updated:
        updated = False
        for p in problems:
            if len(p.vars) == p.k:
                # len(p.vars) > 0, thus p.k > 0
                solutions.update((v, True) for v in p.vars)
                updated = True
            elif p.k == 0:
                solutions.update((v, False) for v in p.vars)
                updated = True
            else:
                unsolved_problems.append(p)
        problems = map(functools.partial(reduce_problem, solutions),
                       unsolved_problems)
        problems = set(filter(None, problems))

    if mproblem is not None:
        mproblem = reduce_problem(solutions, mproblem)
    if mproblem is not None:
        if len(mproblem.vars) == mproblem.k:
            solutions.update((v, True) for v in mproblem.vars)
            mproblem = None
        elif mproblem.k == 0:
            solutions.update((v, False) for v in mproblem.vars)
            mproblem = None
    confidences = {v: 1.0 for v in solutions}
    return solutions, confidences, problems, mproblem


def make_problem_graph(problems, mproblem):
    graph = nx.Graph()
    if mproblem is not None:
        graph.add_node(mproblem, mr=True)
    for p in problems:
        # note that duplicate problems are removed automatically here
        graph.add_node(p, mr=False)
    for u, v in itertools.combinations(graph.nodes, 2):
        # use Jaccard similarity as the connective strength
        vu = frozenset(u.vars)
        vv = frozenset(v.vars)
        jac = len(vu & vv) / len(vu | vv)
        if jac > 0:
            graph.add_edge(u, v, weight=jac)
    return graph


klbisect = nx.algorithms.community.kernighan_lin_bisection


def get_vars(problems):
    return functools.reduce(operator.or_, (set(p.vars) for p in problems))


def solve_problems_graph(graph, max_vars, max_iter, solutions, confidences):
    logger = logging.getLogger(__name__ + '.solve_problems_graph')
    to_remove = None
    at_least_one_nonmproblem = False
    for p, mr in dict(graph.nodes(data='mr')).items():
        if mr and len(p.vars) > max_vars:
            to_remove = p
        if not mr:
            at_least_one_nonmproblem = True
            if to_remove is not None:
                break
    if to_remove and at_least_one_nonmproblem:
        logger.warning('Removed mines_remain problem %s due to exceeding '
                       'n_vars limit (%d > %d)',
                       to_remove, len(to_remove.vars), max_vars)
        graph.remove_node(to_remove)
    for subgraph in map(graph.subgraph, nx.connected_components(graph)):
        workingq = collections.deque([subgraph])
        while workingq:
            head = workingq.popleft()
            n_vars = len(get_vars(head.nodes))
            if n_vars > max_vars:
                logger.warning('Performing Kernighan-Lin bisection due to '
                               'exceeding n_vars limit (%d > %d)',
                               n_vars, max_vars)
                workingq.extend(map(graph.subgraph, klbisect(subgraph)))
            else:
                sols, confs = solve_problems(head.nodes, max_iter)
                solutions.update(sols)
                confidences.update(confs)


def _check_validity_trace(vars2problems, trace):
    """
    Returns ``False`` if not valid; returns ``True`` if valid or not sure.
    """
    trace = dict(trace)
    involved_problems = functools.reduce(operator.or_,
                                         (set(vars2problems[v]) for v in
                                          trace), set())
    for p in involved_problems:
        try:
            if sum(trace[v] for v in p.vars) != p.k:
                return False
        except KeyError:
            pass
    return True


def solve_problems(problems, max_iter):
    logger = logging.getLogger(__name__ + '.solve_problems')
    vars2problems = {v: [p for p in problems if v in p.vars]
                     for v in get_vars(problems)}
    varlist = sorted(vars2problems)

    candidate_solutions = []
    to_open = collections.deque([[]])
    n_iter = 0
    while to_open:
        top_trace = to_open.pop()
        # check if top_trace has already violated certain rules
        validity = _check_validity_trace(vars2problems, top_trace)
        n_iter += 1
        if n_iter >= max_iter:
            logger.warning('Break DFS iterations due to exceeding the '
                           'upper limit %d', max_iter)
            break
        if not validity:
            continue
        if len(top_trace) < len(varlist):
            to_open.append(top_trace + [(varlist[len(top_trace)], True)])
            to_open.append(top_trace + [(varlist[len(top_trace)], False)])
        else:
            candidate_solutions.append([x[1] for x in top_trace])
    candidate_solutions = np.array(candidate_solutions)
    probs = np.sum(candidate_solutions, axis=0) / candidate_solutions.shape[0]

    # FIXME `candidate_solutions` is likely to be [], causing NaN here
    solutions = dict(zip(varlist, map(bool, probs >= 0.5)))
    confidence = dict(zip(varlist, np.maximum(probs, 1 - probs).tolist()))
    return solutions, confidence


def solve(board, mines_remain):
    logger = logging.getLogger(__name__ + '.solve')
    problems, mproblem = encode_board_as_nckproblems(board, mines_remain)
    if not any((problems, mproblem)) or np.all(board == CID['q']):
        logger.info('Performing random guess due to lack to NCK constraints')
        randbloc = np.unravel_index(np.random.randint(board.size), board.shape)
        return np.concatenate((randbloc, [0]))[np.newaxis]

    solutions, confidences, problems, mproblem = \
        simple_solve_attempt(problems, mproblem)
    pgraph = make_problem_graph(problems, mproblem)
    solve_problems_graph(pgraph, MAX_VARS, MAX_ITER, solutions, confidences)
    varlist = list(solutions)
    qidx = np.stack(np.unravel_index(varlist, board.shape), axis=1)
    mine = np.array([solutions[v] for v in varlist], dtype=np.int64)
    qidx_mine = np.concatenate((qidx, mine[:, np.newaxis]), axis=1)
    confidences = np.array([confidences[v] for v in varlist])
    uscore = 1.0 - 1e-6
    if np.max(confidences) > uscore:
        return qidx_mine[np.where(confidences > uscore)]
    return qidx_mine[np.argmax(confidences)][np.newaxis]


def _main():
    args = sutils.make_parser().parse_args()
    try:
        try:
            board, mines_remain = sutils.read_board(args.board_csv)
        except sutils.EmptyCsvError:
            print('EmptyCsvError', file=sys.stderr)
            sys.exit(4)
        qidx_mine = solve(board, mines_remain)
        np.savetxt(sys.stdout, qidx_mine, delimiter=',', fmt='%d')
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        sys.stderr.close()


if __name__ == '__main__':
    _main()
