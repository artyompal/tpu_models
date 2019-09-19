
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_train', help='output file mask, e. g. train_fold_{}.csv', type=str)
    parser.add_argument('output_val', help='output file mask, e. g. val_fold_{}.csv', type=str)
    parser.add_argument('input', help='description file, csv format', type=str)
    parser.add_argument('--num_folds', help='number of parts to split', default=5, type=int)
    args = parser.parse_args()

    print('reading csv')
    df = pd.read_csv(args.input)
    print('df.shape', df.shape)

    classes = df.LabelName.unique()
    class2idx = {c: i for i, c in enumerate(classes)}

    def one_hot_encode(labels: np.array) -> np.array:
        res = np.zeros_like(classes, dtype=float)
        for L in labels:
            res[class2idx[L]] = 1
        return res

    print('grouping samples')
    x = [sample_df for _, sample_df in tqdm(df.groupby('ImageID'), total=df.ImageID.nunique())]

    print('transforming labels')
    y = [one_hot_encode(sample_df.LabelName) for sample_df in tqdm(x)]

    print('splitting the data')
    split = MultilabelStratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=777)

    total = 0

    for fold, (train_idx, val_idx) in enumerate(tqdm(split.split(x, y), total=args.num_folds)):
        print(train_idx, val_idx)

        assert train_idx.size + val_idx.size == len(x)
        total += val_idx.size

        df = pd.concat([x[idx] for idx in train_idx])
        print('df.shape', df.shape)
        df.to_csv(args.output_train.format(fold), index=False)

        df = pd.concat([x[idx] for idx in val_idx])
        print('df.shape', df.shape)
        df.to_csv(args.output_val.format(fold), index=False)

    assert total == len(x)
