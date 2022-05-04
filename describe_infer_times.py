import argparse
import os

import numpy as np
import pandas as pd


def make_parser():
    parser = argparse.ArgumentParser(
        description=('Generate statistical description on a series of log '
                     'analysis.'))
    parser.add_argument('outfile', help='text file to write')
    parser.add_argument('indir', help='directory where log analysis lie')
    parser.add_argument('names', nargs='+', help='the names of the inputs')
    return parser


def calc_stats(indir, names):
    filenames = [os.path.join(indir, x + '.txt') for x in names]
    means = []
    stds = []
    q0s = []
    q25s = []
    q50s = []
    q75s = []
    q100s = []
    for filename in filenames:
        arr = np.loadtxt(filename)
        means.append(np.mean(arr))
        stds.append(np.std(arr))
        q0s.append(np.percentile(arr, 0))
        q25s.append(np.percentile(arr, 25))
        q50s.append(np.percentile(arr, 50))
        q75s.append(np.percentile(arr, 75))
        q100s.append(np.percentile(arr, 100))
    df = pd.DataFrame(
        {
            'mean': means,
            'std': stds,
            'min': q0s,
            'q25': q25s,
            'median': q50s,
            'q75': q75s,
            'max': q100s,
        },
        index=names
    )
    return df


def main():
    args = make_parser().parse_args()
    df = calc_stats(args.indir, args.names)
    df.to_csv(args.outfile, index_label='solver')


if __name__ == '__main__':
    main()
