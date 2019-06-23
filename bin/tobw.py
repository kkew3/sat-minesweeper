#!/usr/bin/env python3
import os
import logging
import argparse
import sys

import PIL.Image as Image

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--suffix', nargs='?', const='')
    parser.add_argument('filenames', nargs='*')
    return parser

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = make_parser().parse_args()
    for filename in args.filenames:
        name, ext = os.path.splitext(filename)
        tofile = ''.join((name, args.suffix, ext))
        try:
            Image.open(filename) \
                 .convert('L') \
                 .point(lambda p: p >= 128 and 255) \
                 .save(tofile)
        except:
            logging.exception()
        else:
            logging.info('Converted %s -> %s', filename, tofile)
