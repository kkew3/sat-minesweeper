import sys
import logging

import networkx as nx
import numpy as np

import guess
import solverutils as sutils
from solverutils import CID
import fullsatsolver
import mcdfssolver

DEFAULT_MAX_VARS = 32


def sat_solve_problems(problems, solver='minisat22', max_solutions=10000):
    logger = logging.getLogger(__name__ + '.sat_solve_problems')

    # encode as CNF clauses
    qvars_to_use = sorted(mcdfssolver.get_vars(problems))
    qvar2vid = {v: i for i, v in enumerate(qvars_to_use, start=1)}
    pe = fullsatsolver.NCKProblemEncoder(len(qvars_to_use))
    clauses = set()
    for p in problems:
        vars_ = sorted(p.vars)
        vars__ = [qvar2vid[v] for v in vars_]
        logger.debug('Translated vars from %s to %s', vars_, vars__)
        C = pe(vars__, p.k)
        logger.debug('Encoded k=%d vars=%s as clauses C=%s', p.k, vars__, C)
        clauses.update(map(tuple, C))
    clauses = list(map(list, clauses))
    logger.debug('Involved vars=%s, final clauses=%s', qvars_to_use, clauses)

    # solve the SAT problem
    solutions = fullsatsolver.attempt_full_solve(clauses, solver,
                                                 max_solutions)

    # analyze solutions
    confidence, mines = fullsatsolver.analyze_solutions(
        solutions, len(qvars_to_use))
    logger.debug('Full solve solutions=%s, confidence=%s', mines.tolist(),
                 confidence.tolist())

    # pack mines and confidence into dict
    qvar2mine = dict(zip(qvars_to_use, mines))
    qvar2confidence = dict(zip(qvars_to_use, confidence))
    return qvar2mine, qvar2confidence


def solve_problems_graph(graph, solutions, confidences, max_vars) -> None:
    """
    :param graph: the problem graph
    :param solutions: dict to put solutions in
    :param confidences: dict to put confidences in
    :param max_vars: till when to stop partitioning problems
    """
    logger = logging.getLogger(__name__ + '.solve_problems_graph')
    workingq = list(map(graph.subgraph, nx.connected_components(graph)))
    while workingq:
        top_graph = workingq.pop()
        n_vars = len(mcdfssolver.get_vars(top_graph.nodes))
        if n_vars > max_vars and len(top_graph) >= 2:
            logger.warning(
                'Performing Min-cut bisection due to '
                'exceeding n_vars limit (%d > %d)', n_vars, max_vars)
            # NodeTooLessError shouldn't be raised unless MAX_VARS is too small
            workingq.extend(
                map(graph.subgraph, mcdfssolver.mincut_bisect(top_graph)))
        else:
            if n_vars > max_vars:
                logger.warning(
                    'Exceeding n_vars limit (%d > %d) but top_graph'
                    ' has only %d node left; stopped bisection', n_vars,
                    max_vars, len(top_graph))
            # top_graph.nodes, i.e. a set of problems.
            # top_graph can't be an empty graph, as mincut_bisect won't output
            # empty graph.
            # there must be at least one node in top_graph
            try:
                sols, confs = sat_solve_problems(top_graph.nodes)
            except sutils.NoSolutionError:
                logger.debug(
                    'NoSolutionError during dfs_solve_problems, with '
                    'top_graph.nodes: %s; suppressed', top_graph.nodes)
            solutions.update(sols)
            confidences.update(confs)


def solve_board(board, mines_remain, max_vars):
    logger = logging.getLogger(__name__ + '.solve_board')
    problems, mproblem = mcdfssolver.encode_board(board, mines_remain)
    logger.debug('Encoded board: %s; %s', problems, mproblem)
    if not problems and not mproblem:
        raise sutils.NoSolutionError
    solutions, confidences, problems, mproblem = \
            mcdfssolver.trivial_solve_attempt(problems, mproblem)
    logger.debug(
        'Trivial solve complete with (partial) solutions: %s; '
        'confidences: %s', solutions, confidences)
    logger.debug('(Possibly) reduced encoding: %s; %s', problems, mproblem)
    pgraph = mcdfssolver.make_problem_graph(problems, mproblem)
    if not pgraph and not solutions:
        raise sutils.NoSolutionError
    solve_problems_graph(pgraph, solutions, confidences, max_vars)
    logger.debug('Graph solve complete with solutions: %s; confidences: %s',
                 solutions, confidences)
    if not solutions:
        raise sutils.NoSolutionError
    varlist = list(solutions)
    qidx = np.stack(np.unravel_index(varlist, board.shape), axis=1)
    mine = np.array([solutions[v] for v in varlist], dtype=np.int64)
    qidx_mine = np.concatenate((qidx, mine[:, np.newaxis]), axis=1)
    confidences = np.array([confidences[v] for v in varlist])
    return qidx_mine, confidences


def solve(board,
          mines_remain,
          consider_mines_th: int = 5,
          guess_edge_weight: float = 2.0,
          max_vars: int = DEFAULT_MAX_VARS,
          _first_bloc=None):
    logger = logging.getLogger(__name__ + '.solve')
    if np.all(board == CID['q']):
        logger.info('Performing first step random guess')
        if _first_bloc:
            randbloc = _first_bloc
        else:
            randbloc = guess.global_uniform(board)
        logger.info('Choosing bloc=%s', randbloc)
        return np.concatenate((randbloc, [0]))[np.newaxis]
    if np.all(board != CID['q']):
        logger.warning('No uncovered cells found. Why isn\'t the game ended?')
        return np.array([])

    try:
        logger.info('Performing Min-cut SAT inference')
        qidx_mine, confidences = solve_board(board, None, max_vars)
        uscore = 1.0 - 1e-6
        if np.max(confidences) <= uscore and mines_remain is not None \
                and mines_remain <= consider_mines_th \
                and np.sum(board == CID['q']) <= max_vars:
            logger.info('No confident decision. Rerunning inference using '
                        'mines_remain')
            qidx_mine, confidences = solve_board(board, mines_remain, max_vars)
        if np.max(confidences) > uscore:
            logger.debug('There exists confidences == 1; use them')
            return qidx_mine[np.nonzero(confidences > uscore)]
        if not np.allclose(np.max(confidences), 0.0):
            logger.debug('There exists no confidence == 0; use max confidence')
            return qidx_mine[np.argmax(confidences)][np.newaxis]
        # confidence == [0.0, 0.0, ...], mines should be [False, False, ...]
        assert not np.any(qidx_mine[:, 2]), qidx_mine
        logger.info('Confidences are all zero; failing back to random guess')
        rand_bloc = guess.prefer_edge(board, qidx_mine[:, :2],
                                      guess_edge_weight)
        logger.info('Choosing: bloc=%s, mine_under=0', rand_bloc)
        return np.concatenate((rand_bloc, [0]))[np.newaxis]
    except sutils.NoSolutionError:
        logger.warning('NoSolutionError')
        logger.info('Falling back to random guess')
        # guess edges with more probability
        all_blocs = np.stack(np.nonzero(board == CID['q']), axis=1)
        rand_bloc = guess.prefer_edge(board, all_blocs, guess_edge_weight)
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
