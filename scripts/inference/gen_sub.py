#!/usr/bin/python3.6

import argparse
import json
import os
import pickle

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Set

import numpy as np
import pandas as pd
import tensorflow as tf

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

def format_prediction(class_id: int, box: np.array, score: float) -> str:
    class_name = classes[class_id - 1]
    box_str = ' '.join(map(str, [box[1], box[0], box[3], box[2]]))
    predicts = [f'{class_name} {score} {box_str}']

    for parent in parents[class_name]:
        predicts.append(f'{parent} {score} {box_str}')

    return ' '.join(predicts)

def format_sample(image_id: str, pred: Any) -> Dict[str, Any]:
    classes = pred['detection_classes']
    boxes = pred['detection_boxes']
    scores = pred['detection_scores']

    classes = classes.numpy()[0]
    boxes = boxes.numpy()[0]
    scores = scores.numpy()[0]

    prediction_strings = [format_prediction(cls, box, score)
                          for cls, box, score in zip(classes, boxes, scores)]

    return {
        'ImageID': image_id,
        'PredictionString': ' '.join(prediction_strings)
    }

def detect_classes_set(filename: str) -> str:
    ''' Finds proper classes.csv for the given prediction. '''
    classes = {
        'human_parts': 'human_parts',
        'leaf_443': 'leaf_443',
        'part_0': 'part_0_of_5',
        'part_1': 'part_1_of_5',
        'part_2': 'part_2_of_5',
        'part_3': 'part_3_of_5',
        'part_4': 'part_4_of_5',
    }

    filename = os.path.basename(filename)

    for key, val in classes.items():
        if key in filename:
            return f'../output/classes_{val}.csv'

    assert False, 'could not find proper classes.csv'

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='object detection model results', type=str)
    args = parser.parse_args()

    classes_csv = detect_classes_set(args.filename)
    classes_df = pd.read_csv(classes_csv, header=None)
    classes = classes_df.iloc[:, 0].values

    parents = load_parents()

    sample_submission_df = pd.read_csv('../data/OBJDET_sample_submission.csv')
    image_ids = sample_submission_df['ImageId']

    tf.enable_eager_execution()

    with open(args.filename, 'rb') as f:
        predictions = pickle.load(f)

    predictions = [format_sample(img, pred) for img, pred in zip(image_ids, tqdm(predictions))]

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.splitext(os.path.basename(args.filename))[0] + '.csv',
                          index=False)
