''' Universal inference script for a saved TF model. '''

import argparse
import os
import pickle

import tensorflow as tf
import pandas as pd

from glob import glob
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='TF SavedModel directory', type=str)
    args = parser.parse_args()

    model_dir = args.directory[args.directory.find('best_models'):]
    model_dir = glob(model_dir + '/*')[0]
    dirs = os.path.normpath(model_dir).split('/')
    assert len(dirs) > 1
    dest_filename = dirs[-2] + '.pkl'

    print('model_dir', model_dir, 'dest_filename', dest_filename)

    tf.enable_eager_execution()
    model = tf.saved_model.load_v2(model_dir, ['serve'])

    signature = model.signatures["serving_default"]
    print('supported outputs:', signature.structured_outputs)

    sample_submission_df = pd.read_csv('data/OBJDET_sample_submission.csv')
    image_ids = sample_submission_df['ImageId']

    predictions = []
    for image_id in tqdm(image_ids):
        img_filename = f'data/test/{image_id}.jpg'

        with open(img_filename, 'rb') as f:
            data = f.read()
            pred = signature(tf.constant(data))
            predictions.append(pred)

    with open(dest_filename, 'wb') as f:
        pickle.dump(predictions, f)
