#!/usr/bin/env python3
import sys

with open(sys.argv[1]) as infile:
    for i, line in enumerate(infile):
        if not i:
            print(line, end='')
        else:
            tokens = sorted(set(map(int, line.rstrip().split())), key=abs)
            print(' '.join(map(str, tokens)))
