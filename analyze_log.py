import argparse
import datetime
import re

import numpy as np


def make_parser():
    parser = argparse.ArgumentParser(
        description='Parse logs/run.log for inference time histogram.')
    parser.add_argument('to_path', help='the output path')
    parser.add_argument(
        '--log',
        default='logs/run.log',
        help='the log file to parse, default to %(default)s')
    return parser


def parse_line(line):
    line = line.rstrip()
    _, dt, _, _, msg = line.split('|')
    dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S,%f')
    return dt, msg


def main():
    args = make_parser().parse_args()
    inference_lasts = []
    with open(args.log, encoding='utf-8') as infile:
        for line in infile:
            dt, msg = parse_line(line)
            if re.match(r'Performing .* inference', msg):
                inference_lasts.append(dt)
            elif re.match(r'Inference done', msg):
                prev_dt = inference_lasts.pop()
                assert isinstance(prev_dt, datetime.datetime), prev_dt
                inference_lasts.append((dt - prev_dt).total_seconds())

    inference_lasts = np.asarray(inference_lasts)
    np.savetxt(args.to_path, inference_lasts, encoding='utf-8')


if __name__ == '__main__':
    main()
