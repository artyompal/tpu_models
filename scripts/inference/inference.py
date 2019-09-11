''' Universal inference script for a saved TF model. '''

import argparse
import os
import pickle

import tensorflow as tf
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', help='result filename, Pickle format', type=str)
    parser.add_argument('directory', help='TF SavedModel directory', type=str)
    args = parser.parse_args()

    tf.enable_eager_execution()
    model = tf.saved_model.load_v2(args.directory, ['serve'])

    signature = model.signatures["serving_default"]
    print('supported outputs:', signature.structured_outputs)

    sample_submission_df = pd.read_csv('data/OBJDET_sample_submission.csv')
    image_ids = sample_submission_df['ImageId']

    predictions = []
    for image_id in tqdm(image_ids):
        filename = f'data/test/{image_id}.jpg'

        with open(filename, 'rb') as f:
            data = f.read()
            pred = signature(tf.constant(data))
            predictions.append(pred)

    with open(args.destination, 'wb') as f:
        pickle.dump(predictions, f)
