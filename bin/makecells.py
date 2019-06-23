#!/usr/bin/env python3
import sys
import os
import logging
import numpy as np
import PIL.Image as Image
import vboard as vb

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    kl = vb.detect_key_lines()
    for filename in sys.argv[1:]:
        name = os.path.basename(filename)
        try:
            mid = int(name[4:name.index('.')])
        except ValueError:
            logging.exception('')
            continue
        basedir = os.path.dirname(os.path.realpath(filename))
        todir = os.path.join(basedir, f'cells{mid}')
        try:
            os.mkdir(todir)
        except FileExistsError:
            logging.exception('')
            continue
        img = np.asarray(Image.open(filename).convert('L'))
        cells, sh = vb.partition_board(kl, img)
        for i, x in enumerate(map(Image.fromarray, cells)):
            x.save(os.path.join(todir, f'c{i:0>3}.png'))
        logging.info('Made %s', os.path.basename(todir))
