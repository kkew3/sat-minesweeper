# SAT-based Classic MineSweeper Solver

![small-demo](demo/small.gif)

![large-demo](demo/large.gif)

## Introduction

This is a SAT-based MineSweeper solver.
It interactively plays online MineSweeper game provided by [freeminesweeper.org](http://www.freeminesweeper.org/minecore.html) through usual human-computer interface.
It detects the board by sliding window template matching and uncover the mines by simulating left/right mouse clicks.
It can play on all configuration of boards (height, width, number of mines).
It logs all events needed to reproduce the process.
It automatically begins a new round whenever it loses or wins.
And it records the winning rate.

## How to use

Open [freeminesweeper.org](http://www.freeminesweeper.org/minecore.html)'s minesweeper board page.
Be sure to expose the entire board on screen.
Now run command

```bash
python3 -m virtualenv rt
. rt/bin/activate
pip install -r requirements.txt

python mwsolver.py

# or perform a one-step solution given a board
#python fullsatsolver.py example_boards/2.csv
```

Again, be sure not to overlap the Terminal window with the board.
Wait 10 seconds the computer will play on its own.

Enjoy watching it playing!

## Mechanism

The core mechanism is to convert current MineSweeper board into a CNF, and resort to [PicoSAT](http://fmv.jku.at/picosat/), a fast SAT solver, to get the solution (Now I use [MiniSAT](http://minisat.se/) instead).
Due to lack of assumptions (e.g. certain key cells haven't been uncovered yet), or due to the limits on the number of CNF clauses (up to 10 million), it often occurs that a number of possible solutions are returned (up to a thousand).
To disambiguate, I find the symbol that maintain its set/clear state the most throughout all solutions.
If that symbol is always set or always cleared, a.k.a. having mine underneath or otherwise, then it's definitely that state.
If that symbol is mostly one state but sometimes the other, then the former state is a better guess.
Although it's the best guess one can make, still it sometimes loses.

How to list CNF clauses is straight forward.
Let x1, x2, x3 be three cells surrounding a cell labeled `2`.
Then x1+x2+x3=2.
Since there is either mine under a cell or no mine under the cell, the three variables are either 0 or 1.
Thus we may enumerate all possible assignments to x1, x2, x3, and write them as DNF clauses.
After that, we may convert DNF to CNF.
Although it's NP-complete, we can [precompute DNF-to-CNF templates](data/MakeCNFTable.java) and look up the template library in runtime, of which the time consumption is negligible.

In [`fullsatsolver.py`](fullsatsolver.py), I use a form of SAT encoding of cardinality constraints, greatly reducing the number of CNF clauses, removing the need to precompute DNF-to-CNF conversions, yet at the expense of introducing some auxiliary boolean variables.
I take into account the number of mines remaining as soon as it's no longer intractable, giving higher accuracy.

## Future works

- An assumption is not taken into consideration, that all remaining uncovered cells have in total `M` mines, where `M` is the number appearing in the upper left of the board.
  Since it changes all the time, recognizing automatically, if not impossible, would be more difficult than template matching.
  Missing such assumption sometimes makes the solver guess poorly in the very last few steps of a round.
- [Patterns](http://www.minesweeper.info/wiki/Strategy) used by human experts are not taken into consideration.
  These patterns are often not derived from current board and the MineSweeper's rules, e.g. the `1-2-1` pattern.
  Since this work requires some tedious pattern recognition code, I didn't put it into this first release.
