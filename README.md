# Local Texture Estimator for Implicit Representation Function

This repository contains the official implementation for LTE introduced in the following paper:

[**Local Texture Estimator for Implicit Representation Function**](https://ipl.dgist.ac.kr/LTE_cvpr.pdf) (CVPR 2022)


## Installation

Our code is based on Ubuntu 20.04, pytorch 1.10.0, CUDA 11.3 (NVIDIA RTX 3090 24GB, sm86) and python 3.6.

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```
conda env create --file environment.yaml
conda activate lte
```


## Quick Start

### 1. Download pre-trained models.

Model|Download
:-:|:-:
EDSR-baseline-LTE|[Google Drive](https://drive.google.com/file/d/108-wQJOTR41JNn_2Q-5X4p07DvgrBNSB/view?usp=sharing)
EDSR-baseline-LTE+|[Google Drive](https://drive.google.com/file/d/1k_BWZWC4tvWA0WouViHAicdTg0pHBp-W/view?usp=sharing)
RDN-LTE|[Google Drive](https://drive.google.com/file/d/1fdj5cvSopIqFi74x9rofPP9O_2HfSp7K/view?usp=sharing)
SwinIR-LTE|[Google Drive](https://drive.google.com/file/d/1DnrL86pUKwRXNLOxoK_GJdrP6IZ3y9nH/view?usp=sharing)

### 2. Reproduce experiments.

**Table 1: EDSR-baseline-LTE**

```bash ./scripts/test-div2k.sh ./save/edsr-baseline-lte.pth 0```

**Table 1: RDN-LTE**

```bash ./scripts/test-div2k.sh ./save/rdn-lte.pth 0```

**Table 1: SwinIR-LTE**

```bash ./scripts/test-div2k-swin.sh ./save/swinir-lte.pth 8 0```

**Table 2: RDN-LTE**

```bash ./scripts/test-benchmark.sh ./save/rdn-lte.pth 0```

**Table 2: SwinIR-LTE**

```bash ./scripts/test-benchmark-swin.sh ./save/swinir-lte.pth 8 0```


## Train & Test

###  **EDSR-baseline-LTE**

**Train**: `python train.py --config configs/train-div2k/train_edsr-baseline-lte.yaml --gpu 0`

**Test**: `python test.py --config configs/test/test-div2k-2.yaml --model save/_train_edsr-baseline-lte/epoch-last.pth --gpu 0`

### **EDSR-baseline-LTE+**

**Train**: `python train.py --config configs/train-div2k/train_edsr-baseline-lte-fast.yaml --gpu 0`

**Test**: `python test.py --config configs/test/test-fast-div2k-2.yaml --fast True --model save/_train_edsr-baseline-lte-fast/epoch-last.pth --gpu 0`

### **RDN-LTE**

**Train**: `python train.py --config configs/train-div2k/train_rdn-lte.yaml --gpu 0,1`

**Test**: `python test.py --config configs/test/test-div2k-2.yaml --model save/_train_rdn-lte/epoch-last.pth --gpu 0`

### **SwinIR-LTE**

**Train**: `python train.py --config configs/train-div2k/train_swinir-lte.yaml --gpu 0,1,2,3`

**Test**: `python test.py --config configs/test/test-div2k-2.yaml --model save/_train_swinir-lte/epoch-last.pth --window 8 --gpu 0`

Model|Training time (# GPU)
:-:|:-:
EDSR-baseline-LTE|21h (1 GPU)
RDN-LTE|82h (2 GPU)
SwinIR-LTE|75h (4 GPU)

We use NVIDIA RTX 3090 24GB for training.


## Fourier Space

The script [Eval-Fourier-Feature-Space](https://github.com/jaewon-lee-b/lte/blob/main/Eval-Fourier-Feature-Space.ipynb) is used to generate the paper plots.


## Demo

`python demo.py --input ./demo/Urban100_img012x2.png --model save/edsr-baseline-lte.pth --scale 2 --output output.png --gpu 0`


## Citation

If you find our work useful in your research, please consider citing our paper:

```
@InProceedings{lte-jaewon-lee,
    author    = {Lee, Jaewon and Jin, Kyong Hwan},
    title     = {Local Texture Estimator for Implicit Representation Function},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {1929-1938}
}
```


## Acknowledgements

This code is built on [LIIF](https://github.com/yinboc/liif) and [SwinIR](https://github.com/JingyunLiang/SwinIR). We thank the authors for sharing their codes.