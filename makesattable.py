import operator as op
from functools import reduce, partial
import itertools
import json


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def dnf(n, k):
    clauses = [list(range(1, n + 1)) for _ in range(ncr(n, k))]
    for i, to_negate in enumerate(itertools.combinations(range(n), n - k)):
        for j in to_negate:
            clauses[i][j] = -clauses[i][j]
    return clauses


def dnf2cnf(dclauses):
    cclauses = set()
    for cc in map(frozenset, itertools.product(*dclauses)):
        for e in cclauses:
            if e <= cc:
                break
        else:
            l = [0 for _ in range(10)]
            for x in cc:
                if l[abs(x)] + x == 0:
                    break
                l[abs(x)] = x
            else:
                cclauses.add(cc)
                continue
    return list(map(partial(sorted, key=abs), cclauses))


def main():
    result = []
    for n in range(1, 9):
        for k in range(0, n + 1):
            if n >= 6 and k not in (0, 1, n - 1, n):
                continue
            result.append([[n, k], dnf2cnf(dnf(n, k))])
    with open('data/sattable.json', 'w') as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()
