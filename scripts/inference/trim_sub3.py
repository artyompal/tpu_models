#!/usr/bin/python3.6

import argparse
import multiprocessing
import os
import sys

from typing import List
from functools import partial

import numpy as np

from tqdm import tqdm


def read_confidences(s: str) -> List[float]:
    return list(map(float, s.split()[7::6]))

def trim_line(threshold: float, s: str) -> str:
    if s.startswith('ImageID'):
        return s

    values = s.split()
    res = []

    for i in range(0, len(values), 6):
        if i < 6 or float(values[i + 1]) > threshold:
            res.extend(values[i : i + 6])

    return ' '.join(res) + '\n'

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='force overwrite', action='store_true')
    parser.add_argument('result', help='result filename', type=str)
    parser.add_argument('filename', help='submission', type=str)
    parser.add_argument('max_num', help='maximum number of predictions', type=int)
    args = parser.parse_args()
    print(args)

    if os.path.exists(args.result) and not args.f:
        print(args.result, 'already exists, exiting')
        sys.exit()

    pool = multiprocessing.Pool()
    print('reading predictions')
    all_confs: List[float] = []

    with open(args.filename) as f:
        for confs in tqdm(pool.imap(read_confidences, f), total=100000):
            all_confs.extend(confs)

    print(f'sorting scores (total {len(all_confs)})')
    assert len(all_confs) >= args.max_num

    pos = len(all_confs) - args.max_num
    threshold = np.partition(all_confs, pos)[pos]
    print('applying threshold', threshold)

    with open(args.filename) as f:
        with open(args.result, 'w') as out:
            for line in tqdm(pool.imap(partial(trim_line, threshold), f), total=100000):
                out.write(line)
