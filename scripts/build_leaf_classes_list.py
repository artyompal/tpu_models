
import argparse
import json
import os
import sys

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_classes', help='500 for objdet, 300 for incseg', type=int)
    args = parser.parse_args()
    assert args.num_classes in [300, 500]

    if args.num_classes == 300:
        filename = 'data/challenge-2019-label300-segmentable-hierarchy.json'
    else:
        filename = 'data/challenge-2019-label500-hierarchy.json'

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

    pd.DataFrame({'classes': leaf_classes}).to_csv(f'classes_leaf_{len(leaf_classes)}.csv',
                                                   index=None)
