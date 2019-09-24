#!/usr/bin/python3.6

import argparse
import multiprocessing
import os
import sys

from tqdm import tqdm


def trim_line(s: str) -> str:
    if s.startswith('ImageID'):
        return s

    values = s.split()

    for i in range(7, len(values), 6):
        if float(values[i]) < args.min_conf:
            return ' '.join(values[:i - 1]) + '\n'

    return s

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('result', help='result filename', type=str)
    parser.add_argument('filename', help='submission', type=str)
    parser.add_argument('min_conf', help='confidence threshold', type=float, default=0.02)
    args = parser.parse_args()

    if os.path.exists(args.result):
        print(args.result, 'already exists, exiting')
        sys.exit()

    pool = multiprocessing.Pool()

    with open(args.filename) as f:
        with open(args.result, 'w') as out:
            for line in tqdm(pool.imap(trim_line, f), total=100000):
                out.write(line)
