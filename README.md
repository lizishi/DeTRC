# Temporal Repetition Counting with Dynamic Action Queries

This is the official PyTorch implementation of the paper *""*


## Installation
We build our code based on the MMaction2 project (1.3.10 version). See [here](https://github.com/open-mmlab/mmaction2) for more details if you are interested.
MMCV is needed before install MMaction2, which can be install with:
```shell
pip install mmcv-full-f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# For example, to install the latest mmcv-full with CUDA 11.1 and PyTorch 1.9.0, use the following command:
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

```
For other CUDA or pytorch version, please refer [here](https://github.com/open-mmlab/mmcv) to get a matched link.


Then, our code can be built by
```shell
cd DeTRC
pip3 install -e .
```

Then, Install the 1D Grid Sampling and RoI Align operators.
```shell
cd DeTRC/model
python setup.py build_ext --inplace
```

## Data preparing
We used the TSN feature of RepCountA and UCFRep datasets, which can be got refer to [here](./docs/prepare_dataset_DeTRC.md)

## Training

Our model can be trained with

```python
python tools/train.py DeTRC/configs/repcount_tsn_feature_enc_contrastive.py --validate
python tools/train.py DeTRC/configs/repcount_tsn_feature_enc_contrastive_mae.py --validate  # for MAE version
```

We recommend to set the `--validate` flag to monitor the training process.

## Test
If you want to test the pretrained model, please use the following code. we provide the pretrained model [here](https://pan.baidu.com/s/1M1qOgytY87KPFOthKpIUDw?pwd=awxe).
```shell
python tools/test.py DeTRC/configs/repcount_tsn_feature_enc_contrastive.py PATH_TO_MODEL_PARAMETER_FILE
python tools/test.py DeTRC/configs/repcount_tsn_feature_enc_contrastive_mae.py PATH_TO_MODEL_PARAMETER_FILE  # for MAE version
```
