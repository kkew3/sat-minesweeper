import typing
import time
import itertools
import logging

import numpy as np
import networkx as nx
import pyautogui as pg

import vboard as vb
import solverutils as sutils


# pylint: disable=too-few-public-methods
class MouseClicker:
    def __init__(self, mon, dpr, bdetector: vb.BoardDetector):
        """
        :param mon: the monitor region
        :param dpr: the device pixel ratio
        :param bdetector: the board detector used
        """
        self.mon = mon
        self.dpr = dpr
        self.bd = bdetector
        self._l = logging.getLogger('.'.join((__name__, type(self).__name__)))

    def do_click(self, ploc: typing.Tuple[int, int], leftbutton: bool):
        """The actual click operation, to be used internally."""
        # the screen location, taking into account device pixel ratio
        sloc = (ploc[0] // self.dpr[0], ploc[1] // self.dpr[1])
        pg.moveTo(sloc[0] + self.mon['left'], sloc[1] + self.mon['top'])
        button = 'left' if leftbutton else 'right'
        pg.click(button=button)

    def click(self, blocs, leftbutton):
        """
        Click various board locations. The actual clicks might be postponed
        until ``commit`` is called.

        :param blocs: board locs
        :param leftbutton: either bool or list of bools
        """
        if isinstance(leftbutton, bool):
            leftbutton = itertools.repeat(leftbutton, len(blocs[0]))
        plocs_x, plocs_y = self.bd.boardloc_as_pixelloc(blocs)
        bx, by = blocs
        for i, (x, y, lb) in enumerate(zip(plocs_x, plocs_y, leftbutton)):
            self.do_click((x, y), lb)
            self._l.info('%s clicked: %s', 'left' if leftbutton else 'right',
                         bx[i], by[i])

    def commit(self):
        """Commit all the clicks."""
        pass

    def click_commit(self, blocs, leftbutton):
        """
        Click and commit.

        :param blocs: ...
        :param leftbutton: ...
        """
        self.click(blocs, leftbutton)
        self.commit()


class LBMouseClicker(MouseClicker):
    """
    MouseClicker that buffers left clicks till commit.
    """
    def __init__(self, mon, dpr, bdetector: vb.BoardDetector, sct):
        """
        :param mon: ...
        :param dpr: ...
        :param bdetector: the ``BoardDetector`` to use
        :param sct: an ``mss.mss`` instance
        """
        super().__init__(mon, dpr, bdetector)
        self.sct = sct
        self.left_bx = np.array([], dtype=int)
        self.left_by = np.array([], dtype=int)

    def click(self, blocs, leftbutton):
        bx, by = blocs
        if isinstance(leftbutton, bool):
            leftbutton = np.array(list(itertools.repeat(leftbutton, len(bx))))
        right_blocs = bx[~leftbutton], by[~leftbutton]
        if np.any(~leftbutton):
            self._l.info('right clicks: %s', list(zip(*right_blocs)))
        for pxy in zip(*self.bd.boardloc_as_pixelloc(right_blocs)):
            self.do_click(pxy, False)
        self.left_bx = np.append(self.left_bx, bx[leftbutton])
        self.left_by = np.append(self.left_by, by[leftbutton])

    def commit(self):
        if self.left_bx.shape[0]:
            board, _, _ = self.bd.recognize_board_and_mr(self.sct)
            values = board[self.left_bx, self.left_by]
            values_diff = np.zeros_like(values, dtype=bool)
            values_diff[values == 0] = True
            blocs = self.left_bx, self.left_by
            self._l.info('left clicks: %s', list(zip(*blocs)))
            for i, pxy in enumerate(zip(*self.bd.boardloc_as_pixelloc(blocs))):
                prev_values = values
                if not values_diff[i]:
                    # NOTE: It's possible to click a recovered cell if it was
                    # recovered as side effects in the previous commit, of
                    # which current commit is unaware. I have no idea how to
                    # fix this right now.
                    self.do_click(pxy, True)
                    board, _, _ = self.bd.recognize_board_and_mr(self.sct)
                    values = board[self.left_bx, self.left_by]
                    values_diff = np.logical_or(prev_values != values,
                                                values_diff)
                    values_diff[values == 0] = True  # could be redundant
                else:
                    self._l.info(
                        'skipped clicking (%d, %d) because '
                        'clicking has no effect', self.left_bx[i],
                        self.left_by[i])
        self.left_bx = np.array([], dtype=int)
        self.left_by = np.array([], dtype=int)


# pylint: disable=too-few-public-methods
class ActionPlanner:
    def __init__(self, delay_after: float, mc: MouseClicker):
        self.mc = mc
        self.delay_after = delay_after
        self._l = logging.getLogger('.'.join((__name__, type(self).__name__)))

    def click_mines(self, board, qidx_mine) -> None:
        """
        Uncover cells using ``MouseClicker`` according to the solutions
        ``qidx_mine``.

        :param board: the board
        :param qidx_mine: of form array([[bx0, by0, m0], [bx1, by1, m1], ...])
               where bxi is the x board coordinate, byi the y board coordinate,
               mi 0 if there's no mine under else 1
        """
        raise NotImplementedError


# pylint: disable=too-few-public-methods
class PlainActionPlanner(ActionPlanner):
    def click_mines(self, _board, qidx_mine):
        blocs = qidx_mine[:, 0], qidx_mine[:, 1]
        self.mc.click_commit(blocs, qidx_mine[:, 2])
        time.sleep(self.delay_after)


# Begin of NoFlagActionPlanner


# pylint: disable=too-few-public-methods
class BoardFlagModifier:
    """
    Used together with any ``NoFlagActionPlanner`` to let the SAT solver aware
    of where all the inferred mines are.
    """
    def __init__(self):
        self.solutions = {}

    def __call__(self, prev_qidx_mine, board):
        if prev_qidx_mine is not None:
            board = np.copy(board)
            for x, y, m in prev_qidx_mine:
                self.solutions[x, y] = m
            for (x, y), m in self.solutions.items():
                if m:
                    board[x, y] = sutils.CID['f']
        return board


# pylint: disable=too-few-public-methods
class NoFlagActionPlanner(ActionPlanner):
    def click_mines(self, _board, qidx_mine):
        no_mine_under = qidx_mine[qidx_mine[:, 2] == 0]
        blocs = no_mine_under[:, 0], no_mine_under[:, 1]
        self.mc.click_commit(blocs, True)
        time.sleep(self.delay_after)


# End of NoFlagActionPlanner
# Begin of GreedyChordActionPlanner


def _locs_from_solutions(qidx_mine):
    """
    Given solutions returns the bi-partition based on whether there are
    mines.
    """
    # covered board locations where there are mines
    flocs = qidx_mine[qidx_mine[:, 2] == 1, :2]
    # covered board locations where there are no mines
    clocs = qidx_mine[qidx_mine[:, 2] == 0, :2]
    # [c]overed [c]ells with [m]ines [loc]ation
    ccm_loc = set(map(tuple, flocs))
    # [c]overed [c]ells [n]o [m]ines [loc]ation
    ccnm_loc = set(map(tuple, clocs))
    return ccm_loc, ccnm_loc


def _calc_chord_stats(iboxof: sutils.IBoxOf, board, ccm_loc, ccnm_loc, th=2):
    """
    :param iboxof: the solverutils.IBoxOf function object to compute box
           indices
    :param board: current board
    :param ccm_loc: the first return value of ``_locs_from_solutions``
    :param ccnm_loc: the second return value of ``_locs_from_solutions``
    :param th: an uncovered number cell must lead to at least this number of
           cells to uncover once chorded; default to 2, by definition
    :return: a dictionary mapping number cells on which to chord to cells
             need to be flagged before chord and cells to be uncovered by
             chord, and a set of indices of covered cells that cannot be
             chorded

    Note that the number of clicks of NF strategy is given by
    ``len(ccnm_loc)``.
    """
    # uncovered number cells to chord
    #   => (set of covered cells to be flagged,
    #       set of covered cells to be chorded)
    ucells_to_chord = {}
    # [c]overed [c]ells with [n]o [m]ines that can [n]ot be [c]horded
    ccnmnc_loc = ccnm_loc.copy()
    for bxy in zip(*np.nonzero((board >= 1) & (board <= 8))):
        # bxy (a 2-tuple) is a candidate uncovered number cell on which to
        # chord
        box_cctf_loc = set()  # covered cells to be flagged
        box_ccnm_loc = set()  # covered cells to be chorded
        box_fc_count = 0  # already flagged cells count
        for box_bxy in zip(*iboxof(bxy)):
            if board[box_bxy] == sutils.CID['f']:
                box_fc_count += 1
            elif box_bxy in ccm_loc:
                box_cctf_loc.add(box_bxy)
            elif box_bxy in ccnm_loc:
                box_ccnm_loc.add(box_bxy)
        if (len(box_cctf_loc) + box_fc_count == board[bxy]
                and len(box_ccnm_loc) >= max(th, len(box_cctf_loc))):
            # chord can be performed on bxy
            ucells_to_chord[bxy] = (box_cctf_loc, box_ccnm_loc)
            ccnmnc_loc -= box_ccnm_loc
    return ucells_to_chord, ccnmnc_loc


def build_graph(ucells_to_chord):
    """
    Build graph modeling the relations between uncovered number cells to
    chord (type_='n'), covered cells with mines that must be flagged before
    the chord (type_='f'), and covered cells without mines that will be
    uncovered by the chord (type_='c'). Each n type_ node connects several
    f type_ nodes and c type_ nodes, but no two n type_ nodes are connected.
    """
    G = nx.Graph()
    G.add_nodes_from(ucells_to_chord, type_='n')
    for n, (fs, cs) in ucells_to_chord.items():
        for f in fs:
            G.add_node(f, type_='f')
            G.add_edge(n, f)
        for c in cs:
            G.add_node(c, type_='c')
            G.add_edge(n, c)
    return G


class ScoreAndCount:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.node_types = G.nodes(data='type_')

    def __call__(self, ucell):
        """
        :param ucell: a key of ucells_to_chord
        """
        score = 0
        to_flag = []
        try:
            for v in self.G.neighbors(ucell):
                if self.node_types[v] == 'f':
                    score -= 1
                    to_flag.append(v)
                # ucell's neighbors' type_ is either 'f' or 'c'
                else:
                    score += 1
            return score, to_flag
        except nx.NetworkXError:
            # ucell has been removed from G
            return None


def find_optimal_ucell(G: nx.Graph, ucells_to_chord):
    """
    Find the optimal ucell with respect to the score; also return the cells
    to flag before chord.
    """
    ucell_nodes = list(ucells_to_chord)
    score_and_count = ScoreAndCount(G)
    result = list(map(score_and_count, ucell_nodes))
    # argmax on scores
    i = np.nanargmax([x[0] if x else np.nan for x in result])
    ucell_star = ucell_nodes[i]
    to_flag_star = result[i][1]
    return ucell_star, to_flag_star


def cleanup_used_nodes(G: nx.Graph, ucell):
    neighbors = list(G.neighbors(ucell))
    for v in neighbors:
        G.remove_node(v)
    G.remove_node(ucell)


def enum_rest_ccells(G: nx.Graph):
    return [
        n for n, type_ in dict(G.nodes(data='type_')).items() if type_ == 'c'
    ]


def find_optimal_chord_strategy(board, qidx_mine, th=2):
    _l = logging.getLogger('{}.find_optimal_chord_strategy'.format(__name__))
    ccm_loc, ccnm_loc = _locs_from_solutions(qidx_mine)
    _l.debug('ccm_loc: %s', ccm_loc)
    _l.debug('ccnm_loc: %s', ccnm_loc)
    iboxof = sutils.IBoxOf(board.shape)
    ucells_to_chord, ccnmnc_loc = _calc_chord_stats(iboxof, board, ccm_loc,
                                                    ccnm_loc, th)
    _l.debug('ucells_to_chord: %s', ucells_to_chord)
    _l.debug('ccnmnc_loc: %s', ccnmnc_loc)
    G = build_graph(ucells_to_chord)
    # each one-step strategy is (ucell_to_chord, fcells_to_flag);
    # the first one-step strategy is NF
    one_step_strategies = [(None, [])]
    # each element is a list of rest ccells;
    # the first element is NF
    all_rest_ccells = [enum_rest_ccells(G)]
    while G:
        ucell_star, to_flag_star = find_optimal_ucell(G, ucells_to_chord)
        cleanup_used_nodes(G, ucell_star)
        one_step_strategies.append((ucell_star, to_flag_star))
        all_rest_ccells.append(enum_rest_ccells(G))
    # aggregated strategies
    agg_strategies = []
    prev_ucells_star = []
    prev_to_flag_star = []
    for ucell_star, to_flag_star in one_step_strategies:
        curr_ucells_star = prev_ucells_star[:]
        if ucell_star:
            curr_ucells_star.append(ucell_star)
        curr_to_flag_star = prev_to_flag_star[:]
        curr_to_flag_star.extend(to_flag_star)
        agg_strategies.append((curr_ucells_star, curr_to_flag_star))
        prev_ucells_star = curr_ucells_star
        prev_to_flag_star = curr_to_flag_star
    all_total_clicks = [
        len(rest_ccells) + len(curr_ucells_star) + len(curr_to_flag_star)
        + len(ccnmnc_loc) for rest_ccells, (
            curr_ucells_star,
            curr_to_flag_star) in zip(all_rest_ccells, agg_strategies)
    ]
    i = np.argmin(all_total_clicks)

    ucells_star, to_flags_star = agg_strategies[i]
    rest_ccells_star = all_rest_ccells[i]
    right_clicks = to_flags_star
    rest_left_clicks = list(itertools.chain(ccnmnc_loc, rest_ccells_star))
    # for debug purpose
    #total_clicks_star = all_total_clicks[i]
    # put ucells_star at first so that when providing GreedyChordActionPlanner
    # with sct, chord will happen first; since chord opens cells, the rest
    # cells may not need to be opened.
    return right_clicks, ucells_star, rest_left_clicks


class GreedyChordActionPlanner(ActionPlanner):
    """
    Note on "Chord": Clicking on numbered/satisfied square will open
    all its neighbors.

    The time complexity of brute-force algorithm to find the optimal chord
    strategy can be at least O(2^n), which is intractable. Therefore, I
    used a greedy algorithm to find a quasi-optimal solution. From experiments,
    the total number of clicks of the greedy algorithm is on average down
    21% with respect to NF strategy.
    """
    def __init__(self, delay_after: float, mc: MouseClicker, sct=None):
        """
        :param delay_after: ...
        :param sct: if provided, should be the ``mss.mss()`` object used to
               provide instant feedback during ``click_mines``
        """
        super().__init__(delay_after, mc)
        self.all_mines_ever_found = set()
        self.mines_flagged = set()
        self.sct = sct

    def expand_partial_solutions(self, qidx_mine):
        """
        Expand the solution so that it contains both cell positions without
        mine underneath and cell positions having been flagged, and keep the
        solution partial so that it does not contain other cell positions.

        :param qidx_mine: the solution
        :return: the expanded partial solution the same form as ``qidx_mine``
        """
        no_mines = qidx_mine[qidx_mine[:, 2] == 0]
        inferred_mines = set(map(tuple, qidx_mine[qidx_mine[:, 2] == 1, :2]))
        self.all_mines_ever_found |= inferred_mines
        mines_not_flagged = self.all_mines_ever_found - self.mines_flagged
        if mines_not_flagged and no_mines.shape[0]:
            mines_not_flagged = np.asarray(list(mines_not_flagged))
            mines_not_flagged = np.append(
                mines_not_flagged,
                np.ones((mines_not_flagged.shape[0], 1), dtype=int),
                axis=1)
            expanded_qidx_mine = np.concatenate([no_mines, mines_not_flagged],
                                                axis=0)
        elif mines_not_flagged:
            mines_not_flagged = np.asarray(list(mines_not_flagged))
            mines_not_flagged = np.append(
                mines_not_flagged,
                np.ones((mines_not_flagged.shape[0], 1), dtype=int),
                axis=1)
            expanded_qidx_mine = mines_not_flagged
        else:
            # it's impossible that both no_mines and mines_not_flagged are
            # empty.
            expanded_qidx_mine = no_mines
        return expanded_qidx_mine

    def click_mines(self, board, qidx_mine):
        expanded_qidx_mine = self.expand_partial_solutions(qidx_mine)
        right_clicks, left_chords, left_clicks = find_optimal_chord_strategy(
            board, expanded_qidx_mine)
        left_clicks = left_chords + left_clicks
        self.mines_flagged |= set(right_clicks)
        if right_clicks:
            right_clicks = np.asarray(right_clicks)
            right_blocs = right_clicks[:, 0], right_clicks[:, 1]
            self.mc.click(right_blocs, False)
        if left_clicks:
            left_clicks = np.asarray(left_clicks)
            left_blocs = left_clicks[:, 0], left_clicks[:, 1]
            self.mc.click(left_blocs, True)
        self.mc.commit()
        time.sleep(self.delay_after)


# pylint: disable=too-few-public-methods
class ChordBoardFlagModifier:
    """
    Used together with ``GreedyChordActionPlanner`` to let the SAT solver
    aware of where all the inferred mines are, while in the same time let the
    action planner know which mines have not been flagged.
    """
    def __init__(self):
        self.solutions = {}
        self.orig_board = None

    def __call__(self, prev_qidx_mine, board):
        self.orig_board = board
        if prev_qidx_mine is not None:
            board = np.copy(board)
            for x, y, m in prev_qidx_mine:
                self.solutions[x, y] = m
            for (x, y), m in self.solutions.items():
                if m:
                    board[x, y] = sutils.CID['f']
        return board

    def rewind_board(self):
        return self.orig_board


# End of GreedyChordActionPlanner
