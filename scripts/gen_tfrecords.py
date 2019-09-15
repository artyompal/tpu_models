# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw Open Images dataset to TFRecord for object_detection.

Example usage:
    python convert_to_tfrecords.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""
import collections

import hashlib
import io
import multiprocessing
import os
import sys

from functools import partial
from typing import Any, Tuple

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import PIL.Image

from tqdm import tqdm

from research.object_detection.utils import dataset_util
from research.object_detection.utils import label_map_util

import tensorflow as tf
flags.DEFINE_boolean('include_masks', False,
                     'Whether to include instance segmentations masks '
                     '(PNG encoded) in the result. default: False.')
flags.DEFINE_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string('image_info_file', '', 'File containing image information')
flags.DEFINE_string('classes_file', '', 'CSV file with allowed classes')
flags.DEFINE_string('output_prefix', '', 'Path to output file')
flags.DEFINE_integer('num_shards', 10, 'Number of shards for output file.')
flags.DEFINE_integer('min_samples_per_class', 0, 'Minimum number of samples per class.')
flags.DEFINE_boolean('display_only', False, 'Don\'t write any file, just show what will be done.')

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

# def create_tf_example(image,
#                       image_dir,
#                       bbox_annotations=None,
#                       category_index=None,
#                       caption_annotations=None,
#                       include_masks=False):

def create_tf_example(group, image2idx):
  """Converts image and annotations to a tf.Example proto.

  Args - OLD DESCRIPTION FOR THE REFERENCE, IGNORE IT.
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    image_dir: directory containing the image files.
    bbox_annotations:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official Open Images dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    category_index: a dict containing Open Images category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    caption_annotations:
      list of dict with keys: [u'id', u'image_id', u'str'].
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_id, image_df = group

  # some settings here
  bbox_annotations = True
  include_masks = False
  image_dir = FLAGS.image_dir

  filename = image_id + '.jpg'

  full_path = os.path.join(image_dir, filename)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()

  pil_image = PIL.Image.open(full_path)
  image_height = pil_image.height
  image_width = pil_image.width

  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image2idx[image_id]).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
  }

  # num_annotations_skipped = 0
  if bbox_annotations:
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    # encoded_mask_png = []

    for ann in image_df.itertuples():
      # (x, y, width, height) = tuple(object_annotations['bbox'])
      # if width <= 0 or height <= 0:
      #   num_annotations_skipped += 1
      #   continue
      # if x + width > image_width or y + height > image_height:
      #   num_annotations_skipped += 1
      #   continue

      # xmin.append(float(x) / image_width)
      # xmax.append(float(x + width) / image_width)
      # ymin.append(float(y) / image_height)
      # ymax.append(float(y + height) / image_height)

      xmin.append(ann.XMin)
      xmax.append(ann.XMax)
      ymin.append(ann.YMin)
      ymax.append(ann.YMax)

      # is_crowd.append(object_annotations['iscrowd'])
      is_crowd.append(False)

      # category_id = int(object_annotations['category_id'])
      category_id = class_indices[ann.LabelName]
      # print(category_id)
      category_ids.append(category_id)

      category_name = class_labels[ann.LabelName].encode('utf8')
      # print(category_name)
      category_names.append(category_name)

      # area.append(object_annotations['area'])
      area.append(abs((ann.XMax - ann.XMin) * (ann.YMax - ann.YMin)))

      # if include_masks:
      #   run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
      #                                       image_height, image_width)
      #
      #   binary_mask = mask.decode(run_len_encoding)
      #
      #   if not object_annotations['iscrowd']:
      #     binary_mask = np.amax(binary_mask, axis=2)
      #
      #   pil_image = PIL.Image.fromarray(binary_mask)
      #   output_io = io.BytesIO()
      #   pil_image.save(output_io, format='PNG')
      #   encoded_mask_png.append(output_io.getvalue())

    feature_dict.update({
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    })

    if include_masks:
      feature_dict['image/object/mask'] = (
          dataset_util.bytes_list_feature(encoded_mask_png))

  # if caption_annotations:
  #   captions = []
  #   for caption_annotation in caption_annotations:
  #     captions.append(caption_annotation['caption'].encode('utf8'))
  #   feature_dict.update({
  #       'image/caption':
  #           dataset_util.bytes_list_feature(captions)})

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example # key, example, num_annotations_skipped


def _load_images_info(images_info_file, classes):
  df = pd.read_csv(images_info_file)

  if classes is not None:
    print('annotations before filtering:', df.shape)
    print(df)

    df = df[df.LabelName.isin(classes)]

    print('annotations after filtering:', df.shape)
    print(df.LabelName.value_counts())

  if FLAGS.min_samples_per_class:
    all_dfs = []
    print('balancing the dataset')
    print('df.shape was', df.shape)

    for class_, class_df in tqdm(df.groupby('LabelName'), total=len(classes)):
      if FLAGS.min_samples_per_class > class_df.shape[0]:
        quot = FLAGS.min_samples_per_class // class_df.shape[0]
        mod = FLAGS.min_samples_per_class % class_df.shape[0]

        all_dfs.extend([class_df] * quot)
        if mod:
            all_dfs.append(class_df.sample(mod))
      else:
        all_dfs.append(class_df)

    df = pd.concat(all_dfs)
    print('df.shape now', df.shape)

    print('annotations after upsampling:', df.shape)
    print(df.LabelName.value_counts())

  if FLAGS.display_only:
    sys.exit()

  # random shuffle
  df = df.sample(frac=1).reset_index(drop=True)
  print(df)
  return df

def _create_tf_record_from_oid_annotations(
    images_info_file,
    image_dir,
    output_path,
    num_shards,
    object_annotations_file=None,
    caption_annotations_file=None,
    include_masks=False,
    classes=None):
  """Loads Open Images annotation csv files and converts to tf.Record format.

  Args:
    images_info_file: CSV file containing image info. The number of tf.Examples
      in the output tf Record files is exactly equal to the number of image info
      entries in this file. This can be any of train/val annotation csv files.
    image_dir: Directory containing the image files.
    output_path: Path to output tfrecord file.
    num_shards: Number of output files to create.
    object_annotations_file: JSON file containing bounding box annotations.
    caption_annotations_file: JSON file containing caption annotations.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    classes: np.array of classes or None
  """

  tf.logging.info('writing to output path: %s', output_path)
  writers = [
      tf.python_io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' %
                                  (i, num_shards)) for i in range(num_shards)
  ]
  annotations = _load_images_info(images_info_file, classes)

  # img_to_obj_annotation = None
  # img_to_caption_annotation = None
  # category_index = None

  # if object_annotations_file:
  #   img_to_obj_annotation, category_index = (
  #       _load_object_annotations(object_annotations_file))

  # if caption_annotations_file:
  #   img_to_caption_annotation = (
  #       _load_caption_annotations(caption_annotations_file))

  # def _get_object_annotation(image_id):
  #   if img_to_obj_annotation:
  #     return img_to_obj_annotation[image_id]
  #   else: return None
  #
  # def _get_caption_annotation(image_id):
  #   if img_to_caption_annotation:
  #     return img_to_caption_annotation[image_id]
  #   else: return None

#   pool = multiprocessing.Pool()
  total_num_annotations_skipped = 0

  unique_ids = sorted(annotations.ImageID.unique())
  image2idx = {image: i for i, image in enumerate(unique_ids)}
  unique_ids_count = len(unique_ids)

#   for idx, tf_example in enumerate(tqdm(pool.imap(partial(create_tf_example, image2idx=image2idx),
#                                                   annotations.groupby('ImageID')),
#                                    total=unique_ids_count)):
#       writers[idx % num_shards].write(tf_example.SerializeToString())

  for idx, df in enumerate(tqdm(annotations.groupby('ImageID'), total=unique_ids_count)):
      tf_example = create_tf_example(df, image2idx=image2idx)
      writers[idx % num_shards].write(tf_example.SerializeToString())


#   for idx, (_, tf_example, num_annotations_skipped) in enumerate(
#       tqdm(pool.imap(_pool_create_tf_example,
#                      annotations.itertuples()))):
#                 [(image,
#                   image_dir,
#                   _get_object_annotation(image['id']),
#                   category_index,
#                   _get_caption_annotation(image['id']),
#                   include_masks)
#                  for image in annotations])):

    # total_num_annotations_skipped += num_annotations_skipped
    # writers[idx % num_shards].write(tf_example.SerializeToString())

#   pool.close()
#   pool.join()

  for writer in writers:
    writer.close()

  tf.logging.info('Finished writing')
  # tf.logging.info('Finished writing, skipped %d annotations.', total_num_annotations_skipped)


def main(_):
  assert FLAGS.image_dir, '"image_dir" is missing.'
  assert FLAGS.output_prefix, '"output_prefix" is missing.'
  assert FLAGS.image_info_file, 'annotation file is missing.'
  assert FLAGS.classes_file, 'classes file is missing.'

  if FLAGS.image_info_file:
    images_info_file = FLAGS.image_info_file

  global classes_df, class_indices, class_labels, classes
  classes_df = pd.read_csv(FLAGS.classes_file, header=None, names=['classes', 'names'])
  class_indices = {row[1]: row[0] + 1 for row in classes_df.itertuples()}
  class_labels = {row[1]: row[2] for row in classes_df.itertuples()}
  classes = classes_df.classes.values


  directory = os.path.dirname(FLAGS.output_prefix)
  if not tf.gfile.IsDirectory(directory):
    tf.gfile.MakeDirs(directory)

  _create_tf_record_from_oid_annotations(
      images_info_file,
      FLAGS.image_dir,
      FLAGS.output_prefix,
      FLAGS.num_shards,
      include_masks=FLAGS.include_masks,
      classes=classes)


if __name__ == '__main__':
  classes_df = None
  class_indices = None
  class_labels = None
  classes = None

  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
