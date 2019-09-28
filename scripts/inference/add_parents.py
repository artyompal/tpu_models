#!/usr/bin/python3.6

import argparse
import json
import multiprocessing
import os
import sys

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set

import numpy as np
import pandas as pd

from tqdm import tqdm


def load_parents() -> DefaultDict[str, Set[str]]:
    ''' Generates a dictionary with child to parent mapping. '''
    def recursive_parse(parents: DefaultDict[str, Set[str]], node: Any) -> None:
        name = node['LabelName']

        if 'Subcategory' in node:
            for child in node['Subcategory']:
                child_name = child['LabelName']

                if name != '/m/0bl9f':
                    parents[child_name].add(name)
                    parents[child_name].update(parents[name])

                recursive_parse(parents, child)

    with open('../data/challenge-2019-label500-hierarchy.json') as f:
        hierarchy = json.load(f)

    parents: DefaultDict[str, Set[str]] = defaultdict(set)
    recursive_parse(parents, hierarchy)

    return parents

def process_line(s: str) -> str:
    if s.startswith('ImageId'):
        return s

    image_id, pred_str = s.split(',')
    preds = np.array(pred_str.split()).reshape(-1, 6)
    res: List[str] = []

    for pred in preds:
        res.extend(pred)
        class_name = pred[0]

        for parent in parents[class_name]:
            res.append(parent)
            res.extend(pred[1:])

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

    parents = load_parents()
    pool = multiprocessing.Pool()

    with open(args.filename) as f:
        with open(args.result, 'w') as out:
            for line in tqdm(pool.imap(process_line, f), total=100000):
                out.write(line)
