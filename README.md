# Efficient Action Counting with Dynamic Queries

<p align="center">
<a href="https://arxiv.org/", target="_blank">
<img src="https://img.shields.io/static/v1?style=for-the-badge&message=arXiv&color=B31B1B&logo=arXiv&logoColor=FFFFFF&label="></a>
<a href="https://shirleymaxx.github.io/action_counting/", target="_blank">
<img src="https://img.shields.io/badge/_-Project-18405a?style=for-the-badge&logo=Google%20Chrome&logoColor=white" alt="Project Page"></a>
<a href="https://www.youtube.com/watch?v=EmBAE9kDHLA&ab_channel=XiaoxuanMa", target="_blank">
<img src="https://img.shields.io/badge/_-Video-ea3323?style=for-the-badge&logo=Youtube&logoColor=white" alt="YouTube"></a>
</p>
This is the official PyTorch implementation of the paper "Efficient Action Counting with Dynamic Queries". It provides a novel perspective to tackle the *Temporal Repetition Counting* problem using a simple yet effective representation for action cycles, reducing the computational complexity from **quadratic** to **linear** with SOTA performance.

<p align="center">
<img src="https://shirleymaxx.github.io/action_counting/images/structure_diff.jpg" style="width: 100%;">
</p>


## Installation
We build our code based on the MMaction2 project (1.3.10 version). See [here](https://github.com/open-mmlab/mmaction2) for more details if you are interested.
MMCV is needed before install MMaction2, which can be install with:
```shell
pip install mmcv-full-f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# For example, to install the latest mmcv-full with CUDA 11.1 and PyTorch 1.9.0, use the following command:
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```
For other CUDA or pytorch version, please refer to [mmcv](https://github.com/open-mmlab/mmcv).


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

## Data
We used the TSN feature of RepCountA and UCFRep datasets. Please refer to the guidance [here](./docs/prepare_dataset_DeTRC.md).

## Train

Our model can be trained with

```python
python tools/train.py DeTRC/configs/repcount_tsn_feature_enc_contrastive.py --validate
python tools/train.py DeTRC/configs/repcount_tsn_feature_enc_contrastive_mae.py --validate  # for MAE version
```

We recommend to set the `--validate` flag to monitor the training process.

## Test
If you want to test the pretrained model, please use the following code. We provide the pretrained model [here](https://pan.baidu.com/s/1M1qOgytY87KPFOthKpIUDw?pwd=awxe).
```shell
python tools/test.py DeTRC/configs/repcount_tsn_feature_enc_contrastive.py PATH_TO_CHECKPOINT
python tools/test.py DeTRC/configs/repcount_tsn_feature_enc_contrastive_mae.py PATH_TO_CHECKPOINT  # for MAE version
```

## Citation

If you find our work useful for your project, please cite the paper as below:

```
@article{li2024efficient,
  title={Efficient Action Counting with Dynamic Queries},
  author={Li, Zishi and Ma, Xiaoxuan and Shang, Qiuyan and Zhu, Wentao and Qiao, Yu and Wang, Yizhou},
  journal={arXiv preprint arXiv:2402.TBD},
  year={2024}
}
```

