# Feature Boosting and Suppression 

Pytorch implementation of [Dynamic Channel Pruning: Feature Boosting and Suppression](https://openreview.net/forum?id=BJxh2j0qYm)

The official tensorflow implementation is released under [https://github.com/deep-fry/mayo](https://github.com/deep-fry/mayo).

## Description
Feature Boosting and Suppression (FBS) is a method that exploits run-time dynamic information flow in CNNs to dynamically prune channel-wise parameters.

![](./images/fbs.png)

## Reproduced Results
| Model | Dataset | MACs Reduction | Paper's Top-1 | Converted Pytorch Top-1 | 
| :----:| :--: | :--:  | :--: | :--:  |
|MCifarNet-50% | CIFAR10 | 3x | 90.54% | 88.50% <sup>*</sup>|

* The accuracy drop may due to the different conv2d behavior when `pad='same'` and `stride=2` between tensorflow and pytorch. 
See this [issue](https://github.com/pytorch/pytorch/issues/3867) for more details.

## Requirements
- pytorch == 1.0
- torchvision

## Setup
- Clone this repo and prepare data
```bash
git clone git@github.com:yulongwang12/pytorch-fbs.git
cd pytorch-fbs
mkdir data.cifar10
```

## Test
- test released MCifarNet model with gating probability = 0.5
```bash
python test_fbs.py --gpu [GPU_ID] --batch_size [TEST_BATCH_SIZE] --ratio 0.5
```
**note**: the model can only achieve best accuracy at `p=0.5`

## Train
TBD
