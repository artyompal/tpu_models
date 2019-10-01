#!/usr/bin/python3.6

import argparse
import itertools
import os
import sys

from typing import List

import pandas as pd

from tqdm import tqdm
import pickle


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('result', help='result filename', type=str)
    parser.add_argument('filenames', help='submissions from partial datasets', nargs='*')
    args = parser.parse_args()
    assert len(args.filenames) > 0

    num_parts = len(args.filenames)
    for i, fname in enumerate(args.filenames):
        assert fname.endswith(f'_part_{i}_of_{num_parts}.pkl')

    all_predictions = []    # type: ignore

    for fname in tqdm(args.filenames):
        with open(fname, 'rb') as f:
            part_pred = pickle.load(f)
            all_predictions.extend(part_pred)

    assert len(all_predictions) == 99999
    
    with open(args.result, 'wb') as f:
        pickle.dump(all_predictions, f)
