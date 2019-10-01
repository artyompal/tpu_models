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

    for i in range(2, len(values), 6):
        for j in range(4):
            values[i + j] = values[i + j][:-1]

    return ' '.join(values) + '\n'

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='force overwrite', action='store_true')
    parser.add_argument('result', help='result filename', type=str)
    parser.add_argument('filename', help='submission', type=str)
    args = parser.parse_args()

    if os.path.exists(args.result) and not args.f:
        print(args.result, 'already exists, exiting')
        sys.exit()

    pool = multiprocessing.Pool()

    with open(args.filename) as f:
        with open(args.result, 'w') as out:
            for line in tqdm(pool.imap(trim_line, f), total=100000):
                out.write(line)
