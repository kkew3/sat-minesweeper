"""
DFS solver based on global Min-cut partition algorithm to reduce recursion
complexity.
"""

import sys
import itertools
import collections
import logging
import functools
import operator

import networkx as nx
import numpy as np

import solverutils as sutils
from solverutils import CID

MAX_VARS = 32
MAX_ITER = 2**24  # takes approximately 1 second

NCKProblem = collections.namedtuple('NCKProblem', 'vars k')


# a placeholder
class NoSolutionError(Exception):
    pass


def boxof(array, center, radius=1):
    return array[max(0, center[0] - radius):center[0] + radius + 1,
                 max(0, center[1] - radius):center[1] + radius + 1]


def encode_board(board, mines_remain):
    """
    Encode the board into a list of ``NCKProblem``s and an optional
    ``NCKProblem`` (mine hint). The problems may not be unique.
    """
    vartable = np.arange(board.size).reshape(board.shape)
    problems = []
    for x, y in zip(*np.nonzero((board >= 1) & (board <= 8))):
        box = boxof(board, (x, y))
        vbox = boxof(vartable, (x, y))
        if np.any(box == CID['q']):
            problems.append(
                NCKProblem(
                    tuple(vbox[box == CID['q']].tolist()),
                    board[x, y]
                    - np.sum((box == CID['m']) | (box == CID['f'])),
                ))
    if mines_remain is not None:
        mproblem = NCKProblem(
            tuple(vartable[board == CID['q']].tolist()), mines_remain)
        assert mproblem.vars, (board, mproblem)
    else:
        mproblem = None
    return problems, mproblem


def reduce_problem(solutions, problem):
    new_vars = tuple(v for v in problem.vars if v not in solutions)
    new_k = problem.k - sum(
        solutions[v] for v in problem.vars if v in solutions)
    assert new_k >= 0, (solutions, problem)
    if not new_vars:
        return None
    return NCKProblem(new_vars, new_k)


def trivial_solve_attempt(problems, mproblem):
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
        problems = (reduce_problem(solutions, p) for p in unsolved_problems)
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
    logger = logging.getLogger(__name__ + '.make_problem_graph')
    graph = nx.Graph()
    if mproblem is not None:
        # mr=True is used to note that this is mine remaining problem
        graph.add_node(mproblem, mr=True)
    for p in problems:
        # note that duplicate problems are removed automatically here
        graph.add_node(p, mr=False)
    for u, v in itertools.combinations(graph.nodes, 2):
        # use Jaccard similarity as the connective strength
        vu = frozenset(u.vars)
        vv = frozenset(v.vars)
        jac = len(vu & vv) / len(vu | vv)
        # jac > 0 means the joint set of vu and vv is nonempty
        if jac > 0:
            # don't change the key name 'weight' -- it's used below in
            # mincut_bisect
            graph.add_edge(u, v, weight=jac)
    logger.debug('Built graph with nodes: %s; edges: %s', graph.nodes,
                 graph.edges(data='weight'))
    return graph


class NodeTooLessError(Exception):
    pass


def mincut_bisect(graph):
    logger = logging.getLogger(__name__ + '.mincut_bisect')
    if len(graph) < 2:
        raise NodeTooLessError
    logger.debug('Bisection start')
    nodes = list(graph.nodes)
    logger.debug('Nodes: %s', nodes)
    u = nodes[0]
    # Reference: https://stanford.edu/~rezab/classes/cme305/W14/Notes/4.pdf
    results = [
        nx.minimum_cut(graph, u, v, capacity='weight') for v in nodes[1:]
    ]
    mincut = min(results, key=lambda x: x[0])
    logger.debug('Mincut result: %s', mincut)
    return mincut[1]


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


def get_vars(problems):
    return functools.reduce(operator.or_, (set(p.vars) for p in problems))


def solve_problems_graph(graph, solutions, confidences) -> None:
    """
    :param graph: the problem graph
    :param solutions: dict to put solutions in
    :param confidences: dict to put confidences in
    """
    logger = logging.getLogger(__name__ + '.solve_problems_graph')
    workingq = list(map(graph.subgraph, nx.connected_components(graph)))
    while workingq:
        top_graph = workingq.pop()
        n_vars = len(get_vars(top_graph.nodes))
        if n_vars > MAX_VARS:
            logger.warning(
                'Performing Min-cut bisection due to '
                'exceeding n_vars limit (%d > %d)', n_vars, MAX_VARS)
            # NodeTooLessError shouldn't be raised unless MAX_VARS is too small
            workingq.extend(map(graph.subgraph, mincut_bisect(top_graph)))
        else:
            # top_graph.nodes, i.e. a set of problems.
            # top_graph can't be an empty graph, as mincut_bisect won't output
            # empty graph.
            # there must be at least one node in top_graph
            try:
                sols, confs = dfs_solve_problems(top_graph.nodes)
            except NoSolutionError:
                logger.debug(
                    'NoSolutionError during dfs_solve_problems, with '
                    'top_graph.nodes: %s; suppressed', top_graph.nodes)
            solutions.update(sols)
            confidences.update(confs)


def _check_validity_trace(vars2problems, trace):
    """
    Returns ``False`` if not valid; returns ``True`` if valid or not sure.
    """
    trace = dict(trace)
    involved_problems = functools.reduce(operator.or_, (set(vars2problems[v])
                                                        for v in trace), set())
    for p in involved_problems:
        try:
            if sum(trace[v] for v in p.vars) != p.k:
                return False
        except KeyError:
            pass
    return True


def dfs_solve_problems(problems):
    """
    Traverse all possibilities by deep first search and reach solutions.
    """
    logger = logging.getLogger(__name__ + '.dfs_solve_problems')
    vars2problems = {
        v: [p for p in problems if v in p.vars]
        for v in get_vars(problems)
    }
    varlist = sorted(vars2problems)

    candidate_solutions = []
    to_open = [[]]
    n_iter = 0
    while to_open:
        top_trace = to_open.pop()
        # check if top_trace has already violated certain rules
        validity = _check_validity_trace(vars2problems, top_trace)
        n_iter += 1
        if n_iter >= MAX_ITER:
            logger.warning(
                'Break DFS iterations due to exceeding the '
                'upper limit %d', MAX_ITER)
            break
        if not validity:
            continue
        if len(top_trace) < len(varlist):
            to_open.append(top_trace + [(varlist[len(top_trace)], True)])
            to_open.append(top_trace + [(varlist[len(top_trace)], False)])
        else:
            candidate_solutions.append([x[1] for x in top_trace])
    if not candidate_solutions:
        # this might happen when the DFS tree is very deep that exceeds
        # MAX_ITER
        raise NoSolutionError
    # now candidate_solutions consists of 1 and -1, where 1 means there's mine
    # and -1 means there's no mine
    candidate_solutions = np.asarray(candidate_solutions) * 2 - 1
    confidence = np.abs(np.sum(candidate_solutions, axis=0)) \
            / candidate_solutions.shape[0]
    confidence = dict(zip(varlist, confidence))
    # if confidence == 0, presume there's no mine so that we can proceed
    solutions = np.sign(np.sum(candidate_solutions, axis=0)) > 0
    solutions = dict(zip(varlist, solutions))
    return solutions, confidence


def solve_board(board, mines_remain: int = None):
    logger = logging.getLogger(__name__ + '.solve_board')
    problems, mproblem = encode_board(board, mines_remain)
    logger.debug('Encoded board: %s; %s', problems, mproblem)
    if not problems and not mproblem:
        raise NoSolutionError
    solutions, confidences, problems, mproblem = \
            trivial_solve_attempt(problems, mproblem)
    logger.debug(
        'Trivial solve complete with (partial) solutions: %s; '
        'confidences: %s', solutions, confidences)
    logger.debug('(Possibly) reduced encoding: %s; %s', problems, mproblem)
    pgraph = make_problem_graph(problems, mproblem)
    if not pgraph and not solutions:
        raise NoSolutionError
    solve_problems_graph(pgraph, solutions, confidences)
    logger.debug('Graph solve complete with solutions: %s; confidences: %s',
                 solutions, confidences)
    if not solutions:
        raise NoSolutionError
    varlist = list(solutions)
    qidx = np.stack(np.unravel_index(varlist, board.shape), axis=1)
    mine = np.array([solutions[v] for v in varlist], dtype=np.int64)
    qidx_mine = np.concatenate((qidx, mine[:, np.newaxis]), axis=1)
    confidences = np.array([confidences[v] for v in varlist])
    return qidx_mine, confidences


def solve(board,
          mines_remain: int = None,
          consider_mines_th: int = 5,
          guess_edge_weight: float = 2.0):
    logger = logging.getLogger(__name__ + '.solve')
    if np.all(board == CID['q']):
        logger.info('Performing first step random guess')
        randbloc = np.unravel_index(np.random.randint(board.size), board.shape)
        logger.info('Choosing bloc=%s', randbloc)
        return np.concatenate((randbloc, [0]))[np.newaxis]
    if np.all(board != CID['q']):
        logger.warning('No uncovered cells found. Why isn\'t the game ended?')
        return np.array([])

    try:
        logger.info('Performing Min-cut DFS inference')
        qidx_mine, confidences = solve_board(board, None)
        uscore = 1.0 - 1e-6
        if np.max(confidences) <= uscore and mines_remain is not None \
                and mines_remain <= consider_mines_th \
                and np.sum(board == CID['q']) <= MAX_VARS:
            logger.info('No confident decision. Rerunning inference using '
                        'mines_remain')
            qidx_mine, confidences = solve_board(board, mines_remain)
        if np.max(confidences) > uscore:
            logger.debug('There exists confidences == 1; use them')
            return qidx_mine[np.nonzero(confidences > uscore)]
        if not np.allclose(np.max(confidences), 0.0):
            logger.debug('There exists no confidence == 0; use max confidence')
            return qidx_mine[np.argmax(confidences)][np.newaxis]
        # confidence == [0.0, 0.0, ...], mines should be [False, False, ...]
        assert not np.any(qidx_mine[:, 2]), qidx_mine
        logger.info('Confidences are all zero; failing back to random guess')
        rand_bloc = guess(board, qidx_mine[:, :2], guess_edge_weight)
        logger.info('Choosing: bloc=%s, mine_under=0', rand_bloc)
        return np.concatenate((rand_bloc, [0]))[np.newaxis]
    except NoSolutionError:
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
        if first_bloc and np.all(board == CID['q']):
            qidx_mine = np.concatenate((first_bloc, [0]))[np.newaxis]
        else:
            qidx_mine = solve(board, mines_remain)
        np.savetxt(sys.stdout, qidx_mine, delimiter=',', fmt='%d')
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        sys.stderr.close()
    return 0


if __name__ == '__main__':
    sys.exit(_main())
