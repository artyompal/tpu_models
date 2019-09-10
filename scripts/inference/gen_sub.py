#!/usr/bin/python3.6

import argparse
import json
import os
import pickle

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Set

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

    with open('data/challenge-2019-label500-hierarchy.json') as f:
        hierarchy = json.load(f)

    parents: DefaultDict[str, Set[str]] = defaultdict(set)
    recursive_parse(parents, hierarchy)

    return parents

def form_one_prediction_string(result: Any, i: int) -> str:
    class_name = result['detection_class_names'][i].decode('utf-8')
    box = result['detection_boxes'][i]
    score = result['detection_scores'][i]

    box_str = ' '.join(map(str, [box[1], box[0], box[3], box[2]]))
    predicts = [f'{class_name} {score} {box_str}']

    if class_name not in classes:
        global skipped
        skipped += 1
        return ''

    for parent in parents[class_name]:
        predicts.append(f'{parent} {score} {box_str}')

    return ' '.join(predicts)

def format_prediction_strings(predictions: Any) -> Dict[str, Any]:
    image_id, result = predictions
    prediction_strings = [form_one_prediction_string(result, i)
                          for i in range(len(result['detection_scores']))]

    return {
        'ImageID': image_id,
        'PredictionString': ' '.join(prediction_strings)
    }

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='object detection model results', type=str)
    args = parser.parse_args()

    classes = pd.read_csv('data/challenge-2019-classes-description-500.csv', header=None)
    classes = set(classes.iloc[:, 0].values)
    assert len(classes) == 500

    parents = load_parents()

    with open(args.filename, 'rb') as f:
        predictions = pickle.load(f)

    skipped = 0
    predictions = [format_prediction_strings(p) for p in tqdm(predictions)]
    print('skipped', skipped)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.splitext(os.path.basename(args.filename))[0] + '.csv',
                          index=False)
