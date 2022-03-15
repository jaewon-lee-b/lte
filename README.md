# LTE
This repository contains the official implementation for LTE introduced in the following paper:

[**Local Texture Estimator for Implicit Representation Function**](https://arxiv.org/abs/2111.08918) (CVPR 2022)


## Quick Start

1. Download a pre-trained model.

Model|Download
:-:|:-:
EDSR-baseline-LTE|[Google Drive](https://drive.google.com/file/d/108-wQJOTR41JNn_2Q-5X4p07DvgrBNSB/view?usp=sharing)
EDSR-baseline-LTE+|[Google Drive](https://drive.google.com/file/d/1k_BWZWC4tvWA0WouViHAicdTg0pHBp-W/view?usp=sharing)
RDN-LTE|[Google Drive](https://drive.google.com/file/d/1fdj5cvSopIqFi74x9rofPP9O_2HfSp7K/view?usp=sharing)
SwinIR-LTE|[Google Drive](https://drive.google.com/file/d/1DnrL86pUKwRXNLOxoK_GJdrP6IZ3y9nH/view?usp=sharing)

2. Reproduce Experiments

**Table 1: EDSR-baseline-LTE**:`bash ./scripts/test-div2k.sh ./save/edsr-baseline-lte.pth 0`

**Table 1: RDN-LTE**:`bash ./scripts/test-div2k.sh ./save/rdn-lte.pth 0`

**Table 1: SwinIR-LTE**:`bash ./scripts/test-div2k-swin.sh ./save/swinir-lte.pth 8 0`

**Table 2: RDN-LTE**:`bash ./scripts/test-benchmark.sh ./save/rdn-lte.pth 0`

**Table 2: SwinIR-LTE**"`bash ./scripts/test-benchmark-swin.sh ./save/swinir-lte.pth 8 0`