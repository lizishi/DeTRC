# Dataset Preparation

## Download Dataset

### Get RepCount Dataset

DownLoad the dataset from [RepCount](https://svip-lab.github.io/dataset/RepCount_dataset.html), and unzip the file.

# Feature Extraction for Training

As there is currently no backbone integration, we need to extract features from the video files.

## Extract Raw Frames

```bash
# new-short resizes the shorter edge to 256
mkdir -p datasets/LLSP/frames
python tools/data/build_rawframes.py /path/to/video/folder/train ./datasets/LLSP/frames/train --level 1 --ext mp4 --task rgb --new-short 256 --use-opencv
python tools/data/build_rawframes.py /path/to/video/folder/valid ./datasets/LLSP/frames/valid --level 1 --ext mp4 --task rgb --new-short 256 --use-opencv
python tools/data/build_rawframes.py /path/to/video/folder/test ./datasets/LLSP/frames/test --level 1 --ext mp4 --task rgb --new-short 256 --use-opencv
```

## Get File Names

```bash
# Modify the paths inside
mkdir -p datasets/LLSP/temp
python tools/data/build_label_list.py ./datasets/LLSP/frames ./datasets/LLSP/temp annt_file
```

## Extract Features

```bash
mkdir checkpoints
wget https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth -P checkpoints

python tools/data/activitynet/tsn_feature_extraction.py --data-list ./datasets/LLSP/temp/annt_file_train.txt --output-prefix ./datasets/LLSP/feature-frame/train/ --modality RGB --ckpt ./checkpoints/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth --frame-interval 1
python tools/data/activitynet/tsn_feature_extraction.py --data-list ./datasets/LLSP/temp/annt_file_test.txt --output-prefix ./datasets/LLSP/feature-frame/test/ --modality RGB --ckpt ./checkpoints/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth --frame-interval 1
python tools/data/activitynet/tsn_feature_extraction.py --data-list ./datasets/LLSP/temp/annt_file_valid.txt --output-prefix ./datasets/LLSP/feature-frame/valid/ --modality RGB --ckpt ./checkpoints/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth --frame-interval 1
```

## Merge into H5

```bash
python React/combine_pkl_to_h5.py
```

# Prepare Training Configuration File

Make a copy of React/config/repcount_tsn_feature.py and make the following modifications:

```bash
# Dataset settings
dataset_type = "RepCountDataset"
data_root_train = "./datasets/LLSP/feature/train_rgb.h5"
data_root_val = "./datasets/LLSP/feature/valid_rgb.h5"
data_root_test = "./datasets/LLSP/feature/test_rgb.h5"
flow_root_train = None
flow_root_val = None

ann_file_train = "./datasets/annotation/train_new.csv"
ann_file_val = "./datasets/annotation/valid_new.csv"
ann_file_test = "./datasets/annotation/test_new.csv"

# Work directory
work_dir = "./tmp"
```
