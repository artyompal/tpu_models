#!/usr/bin/python3.6

import argparse
import itertools
import sys

from typing import List

import pandas as pd

from tqdm import tqdm


def merge(lines: List[str]) -> str:
    preds = []
    step = 6

    for line in lines:
        values = line.split()

        for ofs in range(0, len(values), step):
            preds.append(values[ofs : ofs + step])
            assert len(preds[-1]) == step

    preds.sort(key=lambda pred: -float(pred[1]))
    res = ' '.join(itertools.chain(*preds))
    return res

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('result', help='result filename', type=str)
    parser.add_argument('filenames', help='submissions from partial datasets', nargs='*')
    args = parser.parse_args()
    assert len(args.filenames) > 1

    print('reading predictions')
    predicts = [pd.read_csv(fname) for fname in tqdm(args.filenames)]
    result = predicts[0].copy()

    print('combining predictions')
    for line in tqdm(range(result.shape[0])):
        result.PredictionString[line] = merge([pred.PredictionString[line] for pred in predicts])

    result.to_csv(args.result, index=False)
