#!/usr/bin/python3.6

import argparse
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm


def trim_line(s: str) -> str:
    if s.startswith('ImageId'):
        return s

    image_id, pred_str = s.split(',')
    preds = np.array(pred_str.split()).reshape(-1, 6)
    res = [' '.join(pred) for pred in preds if pred[0] in classes]
    return image_id + ',' + ' '.join(res) + '\n'

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='force overwrite', action='store_true')
    parser.add_argument('result', help='result filename', type=str)
    parser.add_argument('filename', help='submission', type=str)
    args = parser.parse_args()

    classes_df = pd.read_csv('../output/classes_leaf_443.csv', header=None)
    classes = set(classes_df.iloc[:, 0].values)
    assert len(classes) == 443

    if os.path.exists(args.result) and not args.f:
        print(args.result, 'already exists, exiting')
        sys.exit()

    pool = multiprocessing.Pool()

    with open(args.filename) as f:
        with open(args.result, 'w') as out:
            for line in tqdm(pool.imap(trim_line, f), total=100000):
                out.write(line)
