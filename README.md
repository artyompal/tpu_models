# TPU models for Open Images 2019 Object Detection Challenge | Kaggle

### How-to

To get used with TPUs, I recommend these tutorials:
* https://cloud.google.com/tpu/docs/tutorials/resnet for image classification;
* https://cloud.google.com/tpu/docs/tutorials/retinanet for object detection;
* https://cloud.google.com/tpu/docs/tutorials/mask-rcnn for instance segmentation.

### My changes to this repo

1. I added [scripts/](https://github.com/artyompal/tpu_models/tree/master/scripts) directory with my data processing and inference scripts.
2. I added a lot of models into the [models/official/detection/configs/yaml/](https://github.com/artyompal/tpu_models/tree/master/models/official/detection/configs/yaml).
3. I added EfficientNet support into the RetinaNet code: https://github.com/artyompal/tpu_models/blob/master/models/official/detection/modeling/architecture/efficientnet_model.py
4. I also added SE-ResNext support, but it's incomplete and works too slow. At least, you'd have to transpose channels: https://github.com/artyompal/tpu_models/blob/master/models/official/detection/modeling/architecture/seresnext.py
5. [Here](https://github.com/artyompal/tpu_models/commit/c58a31491932b801f31dc9f65894fbd8f492a0a9) I fixed the train_and_eval loop not supporting restarts: https://github.com/tensorflow/tpu/issues/496
6. [Here](https://github.com/artyompal/tpu_models/blob/master/models/official/detection/modeling/base_model.py#L182) I fixed loading of pretrained models. Standard `tf.init_from_checkpoint()` function fails even if a single variable is missing. So it fails with pretrained EfficientNets in case they use different optimizer than you. I provided a more permissive function.


## Training and inference process

### Data preparation

My training pipeline for this complex dataset is quite complex. Maybe you won't need all of this. Anyway, this is what I do.

I remove any annotations with IsGroupOf. Then I split classes into 6 groups by frequency: 1-100, 101-200, 201-300, 301-400, 400-432 and Human Parts (11 classes, a special dataset is provided for these). Then I pick 5 images per class for validation (I handpicked correct samples by blacklisting incorrect ones). Finally, I generate COCO-style validation.json. To achieve all of this, just do:
```
cd scripts/
./prepare_datasets.sh
```

Then I build TFRecord files and upload them to GCS. Replace links to `gs://new_tpu_storage/` with the link to your very own GSC storage and run this:
```
prepare_tfrecords_v1.sh
upload_files_to_gcs.sh 
```


### Create an instance

Open the Cloud Console in your Google Cloud Project webpage and use the `ctpu` tool to provision instances with TPU:
```
gcloud config set project $YOUR_TPU_PROJECT_ID

and then one of:
ctpu up --tpu-size=v3-8 --machine-type n1-standard-4 --zone ZONE --name TPU_NAME
or:
ctpu up --tpu-size=v2-8 --machine-type n1-highmem-2  --zone ZONE --name TPU_NAME
or:
ctpu up --tpu-size=v2-8 --machine-type n1-highmem-2 --zone ZONE --name TPU_NAME --preemptible
or when you TPU has been preempted and you need a new one:
ctpu up --tpu-size=v2-8 --zone ZONE --name TPU_NAME --preemptible --tpu-only --noconf

optional: open the port for TensorBoard
gcloud compute firewall-rules create tensorboard --allow tcp:6006 --source-tags=TPU_NAME --source-ranges=0.0.0.0/0
```


### Train a model

SSH into an instance:
```bash
gcloud beta compute --project YOUR_TPU_PROJECT_ID ssh --zone YOU_ZONE TPU_NAME
```

And then:
```bash
cd tpu_models/scripts/
git pull
./train_on_dataset.sh DATASET_PART VERSION
or
./train_on_fold.sh DATASET_PART VERSION FOLD_NUMBER
```


### Export a model

Saved models are in TPU-specific format. You can export them into a device-independent format like this:
```bash
./export_saved_model.sh PART VERSION STEP
```

### Infer predictions

When you have downloaded you model from the cloud, you can run the inference on your GPU like this. I use Docker to take care of TensorFlow and CUDA, but it's not necessary:
```bash
./docker_run.sh python inference.py PREDICTIONS.PKL MODEL_DIR
```

### Generate submission

When you have a Pickle file with predictions, you can generate a csv submission:
```
# generate a csv file:
./gen_sub.py PREDICTIONS.PKL

# combine csv files from different parts (simply concatenates)
./join_subs.py sub.csv sub_part_0.csv sub_part_1.csv sub_part_2.csv sub_part_3.csv sub_part_4.csv sub_human_parts.csv

# combine predictions from several models (uses Soft-NMS)
./merge_subs.py ensemble.csv model1.csv model2.csv model3.csv ...

# drop predictions below threshold, e.g. 0.03
./trim_sub_by_threshold.py trimmed_sub.csv sub.csv 0.03

# drop prediction by total number, sorting by confidence (maximum for Kaggle scorer is about 150M predictions)
./trim_sub_by_num_of_predicts.py trimmed_sub.csv sub.csv 150000000
```

## Useful tips

### My way of configuring an instance

```bash
sudo apt install -y mc htop python-tk

echo export PYTHONPATH=$HOME/tpu_models/models >>~/.bashrc
export PYTHONPATH=$HOME/tpu_models/models

git config --global core.editor "vim"
git config --global diff.tool "vimdiff"
git config --global --add difftool.prompt false
git config --global user.name "some name"
git config --global user.email "some@name.here"
git config --global alias.alias "config --get-regexp ^alias\."
git config --global alias.lg "log --graph --pretty=format:'%Cgreen(%h) -%Cblue(%ci) %C(yellow)<%an>%d%Creset %s' --abbrev-commit"

pip install --user Cython matplotlib opencv-python-headless pyyaml Pillow
pip install --user 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'

git clone git@github.com:artyompal/tpu_models.git
cd tpu_models/scripts/
```

### Print variables in a TensorFlow checkpoint

```python
import tensorflow as tf

def restore_from_checkpoint(self):
  print([n.name for n in tf.get_default_graph().as_graph_def().node])

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file(CHECKPOINT_PATH, tensor_name='', all_tensors=True)
```

### Monitor training

Using TensorBoard for 10+ instances is pain, so I use Bash scripting with some grepping:

```bash
# print all AP50 scores
cat training.log | grep -oE '(.AP50.:[ .0-9]+|Restoring.*)' | uniq

# take the freshest log file from the current directory
cat `ls -ct1 | head -n 1` | grep -oE '(.AP50.:[ .0-9]+|Restoring.*)' | uniq

# take the freshest log file and print the best score
cat `ls -ct1 | head -n 1` | grep -oE '(.AP50.:[ .0-9]+)' | uniq | sort -r | head -n 1
```

### Training ImageNet model
It works out of the box. Register on ImageNet and use `tools/dataset/imagenet_to_gcs.py` to download the data and convert it to TFRecords. Then run this:

```bash
export DEPTH=101 # 50, 101, 152, 200
python resnet_main.py --tpu=$HOSTNAME --data_dir=GS_SOURCE_STORAGE \
    --model_dir=GS_DESTINATION_STORAGE --resnet_depth=$DEPTH \
    --config_file=configs/resnet$DEPTH.yaml 2>&1 | tee -a ~/resnet$DEPTH.log

# print all validation results
cat resnet$DEPTH.log | grep 'Saving dict' | grep -v INFO

# print the best validation scores
cat resnet$DEPTH.log | grep 'Saving dict' | grep -v INFO | cut -d' ' -f 19- | sort -r | head

# print the best validation score with step number
cat resnet$DEPTH.log | grep `cat resnet$DEPTH.log | grep 'Saving dict' | grep -v INFO | cut -d' ' -f 19 | sort -r | head -n 1` | grep Saving | grep INFO
```
