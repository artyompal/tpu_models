
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output filename, csv format', type=str)
    parser.add_argument('desc_file', help='description file, csv format', type=str)
    parser.add_argument('predict_file', help='predictions file, csv format', type=str)
    parser.add_argument('--min_conf', help='minimum confidence level', default=0.9, type=float)
    parser.add_argument('--classes', help='classes table', default=None, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.desc_file, index_col=None)
    print('df.shape before', df.shape)

    pred_df = pd.read_csv(args.predict_file, index_col=None)
    new_ids, new_labels, new_coords = [], [], []

    classes = None
    if args.classes is not None:
        classes = set(pd.read_csv(args.classes, header=None).iloc[:, 0].values)
        print('classes:', len(classes), classes)

    for row in tqdm(pred_df.itertuples(), total=pred_df.shape[0]):
        items = np.array(row.PredictionString.split())
        items = items.reshape(-1, 6)

        for pred in items:
            label, conf = pred[0], float(pred[1])

            if classes is not None:
                if label not in classes:
                    continue
                
            if conf > args.min_conf:
                coords = np.array(list(map(float, pred[2:])))

                new_ids.append(row.ImageId)
                new_labels.append(label)
                new_coords.append(coords)

    new_coords = np.concatenate(new_coords).T
    new_df = pd.DataFrame({
        'ImageID': new_ids,
        'Source': 'pseudolabels',
        'LabelName': new_labels,
        'Confidence': 1,
        'XMin': new_coords[0],
        'XMax': new_coords[1],
        'YMin': new_coords[2],
        'YMax': new_coords[3],
        'IsOccluded': 0,
        'IsTruncated': 0,
        'IsGroupOf': 0,
        'IsDepiction': 0,
        'IsInside': 0
        })

    print(new_df.head())
    df = pd.concat([df, new_df], sort=False, ignore_index=True)

    print('df.shape after', df.shape)
    print(df)

    df.to_csv(args.output, index=False)
