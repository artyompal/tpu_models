# TPU models for Open Images 2019 Object Detection Challenge | Kaggle

### How-to

To get used with TPUs, I recommend these tutorials:
* https://cloud.google.com/tpu/docs/tutorials/resnet for image classification;
* https://cloud.google.com/tpu/docs/tutorials/retinanet for object detection;
* https://cloud.google.com/tpu/docs/tutorials/mask-rcnn for instance segmentation.

### Create an instance

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


### Debug TensorFlow model

```python
def restore_from_checkpoint(self):
  print([n.name for n in tf.get_default_graph().as_graph_def().node])

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file('/tmp/model.ckpt', tensor_name='', all_tensors=True)
```


### Monitor training

```bash
# print all AP50 scores
cat training.log | grep -oE '(.AP50.:[ .0-9]+|Restoring.*)' | uniq

# take the freshest log file from the current directory
cat `ls -ct1 | head -n 1` | grep -oE '(.AP50.:[ .0-9]+|Restoring.*)' | uniq

# take the freshest log file and print the best score
cat `ls -ct1 | head -n 1` | grep -oE '(.AP50.:[ .0-9]+)' | uniq | sort -r | head -n 1
```


### Export a model

```bash
./export_saved_model.sh PART VERSION STEP
```


### Infer predictions

```bash
CUDA_VISIBLE_DEVICES=0 ./docker_run.sh python inference.py PREDICTIONS.PKL MODEL_DIR
```

### Training ImageNet

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
