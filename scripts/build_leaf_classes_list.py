
import argparse
import json

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from debug import dprint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_classes', help='500 for objdet, 300 for incseg', type=int)
    args = parser.parse_args()
    assert args.num_classes in [300, 500]

    if args.num_classes == 300:
        filename = 'data/challenge-2019-label300-segmentable-hierarchy.json'
        classes_df = pd.read_csv('data/challenge-2019-classes-description-segmentable.csv', header=None)
    else:
        filename = 'data/challenge-2019-label500-hierarchy.json'
        classes_df = pd.read_csv('data/challenge-2019-classes-description-500.csv', header=None)

    dprint(classes_df.shape)
    assert classes_df.shape[0] == args.num_classes

    with open(filename) as f:
        hierarchy = json.load(f)

    leaf_classes = []

    def walk(node: Any) -> None:
        if 'Subcategory' in node:
            for child in node['Subcategory']:
                walk(child)
        else:
            leaf_classes.append(node['LabelName'])

    walk(hierarchy)
    leaf_classes = sorted(set(leaf_classes))
    print('leaf classes found:', len(leaf_classes))

    classes_df = classes_df[classes_df.iloc[:, 0].isin(leaf_classes)]
    classes_df.to_csv(f'output/classes_leaf_{len(leaf_classes)}.csv', header=False, index=False)
