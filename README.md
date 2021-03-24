# RTNet

This is a PyTorch implementation of CVPR2021 paper "Reciprocal Transformations for Unsupervised Video Object Segmentation". [Arxiv]()

## Prerequisites
For the Resnet34 based model, the training is conducted on four GeForce RTX 2080Ti GPUs with 11GB Memory. 
For the Resnext50 based model, the training is conducted on four V100-SXM2 GPUs with 32GB Memory.

* Python 
* PyTorch 1.6.0
* Torchvision 0.7

## Train
### Datasets
In the paper, we use two datasets: [DAVIS16](https://davischallenge.org/) and [DUTS](http://saliencydetection.net/duts/). Note that the images need to be vertically and horizontally flipped and saved, therefore, the number of images is four times as large as that of original dataset.

### Prepare Optical Flow
Please following the the instruction of [RAFT](https://github.com/princeton-vl/RAFT) to prepare the optial flow. Note that both forward and backward optical flow is required. The optical flows are also calculated flipped images instead of flipping the optical flow of the original images.

### Train
Download the pretrained model of appearance (spatial-R34 or spatial RX-50) and motion stream (temporal-R34 or temporal RX-50) in [Goolge Drive](https://drive.google.com/drive/folders/1W3Nk47YQdVYg6oy2NumFYDqGLw3AgWyT?usp=sharing), [Baidu Pan](https://pan.baidu.com/s/1GagzxzzUQfmhLX7jNpYlmw) (code:ohyo) into ```./models```.
The training code of these two streams can also be found there.

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train-distribuetd.py
```

## Test
1. Download pretrained model (model_R34.pth or model_RX50.pth) from [Google Drive](https://drive.google.com/drive/folders/1cSAb0bpMVRihX54E0jYD848qegLiNpSJ?usp=sharing), [Baidu Pan](https://pan.baidu.com/s/1vdLW3aJUF_3CqBIbc_LQ8Q) (code:296x) into ```./saved_model```

2. Run ```python test.py```

## Pre-computed segmentation maps
You can download the pre-computed segmentation maps from [Google Drive](https://drive.google.com/drive/folders/19iCjt4gaj6QRgKjh8X2uGpGRFMJ9HF6w?usp=sharing), [Baidu Pan](https://pan.baidu.com/s/1vBKM3kJhOIysgJWF9Tf-Tw) (code:3tkj)

## Citation
```
@inproceedings{ren2020rtnet,
  title={Reciprocal Transformations for Unsupervised Video Object Segmentation},
  author={Sucheng, Ren and Wenxi, Liu and Yongtuo, Liu and Haoxin, Chen and Guoqiang, Han and Shengfeng, He},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
```
## Conatact 
For any questions, please feel free to contact [Sucheng Ren](mailto:oliverrensut@gmail.com).