# PU-EdgeFormer: Edge Transformer for Dense Prediction in Point Cloud Upsampling

by Dohoon Kim, Minwoo Shin, and Joonki Paik.

This is the official implementation of [PU-Edgeformer: Edge Transformer for Dense Prediction for Point Cloud Upsampling](https://arxiv.org/abs/2305.01148).

This repository supports training our paper PU-EdgeFormer, and previous methods [PU-Net](https://github.com/yulequan/PU-Net), [MPU](https://github.com/yifita/3PU), [PU-GAN](https://github.com/liruihui/PU-GAN), [PU-GCN](https://github.com/guochengqian/PU-GCN).

## Installation
    git clone https://github.com/dohoon2045/PU-EdgeFormer.git
    cd puedgeformer
    bach env_install.sh
    conda activate puedgeformer
    
## Dataset
We use PU1K dataset for training and testing as provided by [PU-GCN](https://github.com/guochengqian/PU-GCN).
Please refer to original repository for downloading the data.

You can also use other dataset of h5 format such as provided by [PU-GAN](https://github.com/liruihui/PU-GAN), [PU-Net](https://github.com/yulequan/PU-Net).

## Training
    python main.py --phase train --model puedgeformer --log_dir log/pu-edgeformer/
    
## Testing
    python main.py --phase test --model puedgeformer --log_dir log/pu-edgeformer/ --data_dir ./data/PU1K/test/input_2048/input_2048/
    
## Evaluation
    python evaluate.py --gt ./data/PU1K/test/input_2048/gt_8192/ --pred evaluation_code/result/ --save_path log/pu-edgeformer/
    
## Citation
If you find PU-EdgeFormer is useful your research, please consider citing:

    @article{kim2023pu,
    title={PU-EdgeFormer: Edge Transformer for Dense Prediction in Point Cloud Upsampling},
    author={Kim, Dohoon and Shin, Minwoo and Paik, Joonki},
    journal={arXiv preprint arXiv:2305.01148},
    year={2023}
    }

## Acknowledgement
This repo is heavily built based on [PU-GCN](https://github.com/guochengqian/PU-GCN) and [PU-GAN](https://github.com/liruihui/PU-GAN) code.
We also borrow the architecture and evaluation codes from [PU-Net](https://github.com/yulequan/PU-Net) and [MPU](https://github.com/yifita/3PU).
