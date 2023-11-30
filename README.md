<h1 align="left">Continuous Pose for Monocular Cameras in Neural Implicit Representation
 <a href="https://arxiv.org/abs/2311.17119"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a> </h1> 

<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#results-demo">Results Demo</a> |
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#news">News</a> |
  <!-- <a href="#statement">Statement</a> |
  <a href="#reference">Reference</a> -->
</p>




## Introduction

This repository contains the example code, test results for the paper Continuous Pose for Monocular Cameras in Neural Implicit Representation. It showcase the effectiveness of optimizing monocular camera poses as a continuous function of time with neural network.

We have released the demo code, more details will be released soon, please check news for details.

## Results-demo
We test our method on multiplte NeRF application, based on amazing code borrowed from [BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF/tree/main), [EventNeRF](https://4dqv.mpi-inf.mpg.de/EventNeRF/) and [NICE-SLAM](https://github.com/cvg/nice-slam). 

<img src="asset/imgs/main_diag_0_v3.png">

We also plan to try it on more application like NeRFstudio, also welcome to give it shot on your application.

### Some of the results
BaRF results of initialize pose, BARF reuslts and our results


<img src="asset/imgs/barf_start.png" width="100"><img src="asset/imgs/barf_their.png" width="100"><img src="asset/imgs/barf_ours.png" width="100">


EventNeRF compare

<img src="asset/imgs/angle28r_00770.png" width="150"><img src="asset/imgs/angle28_ours_r_00770.png" width="150">

NICE-SLAM compare

<img src="asset/imgs/compare_traj_scan0000_start.png" width="150"><img src="asset/imgs/compare_traj_scan0207_start.png" width="150">


## Installation
Regarding the environment you need nothing but numpy, torch to make it work

## Quick start
You can check the code block in the bottom of PoseNet.py to get some idea of how it work. In short:
```python
from PoseNet import PoseNet 

# create a config dict
## if no imu is used, you can set min_time = 0 and max_time to number of images
## remember set activ to softplus if you calculate the derivatives
## max iterations is used to set scheduler
config = {
        'device': "cuda:0", 
        'poseNet_freq': 5,
        'layers_feat': [None,256,256,256,256,256,256,256,256],
        'skip': [4],  
        'min_time': 0,
        'max_time': 100,
        'activ': 'relu',
        'cam_lr': 1e-3,
        'use_scheduler': False
        'max_iter': 20000,
        }

# create instance
posenet = PoseNet(config)

# get a Nx3x4 pose matrix 
time = 1
est_pose = posenet.forward(time)

# add the step function after loss backward
loss.backward() # according to your implementation

posenet.step()

```

# News
- [x] Upload demo code
- [ ] Release pip package (XD new user suspend)
- [ ] Release example on Nerfstudio
- [ ] Release mono-lego dataset
- [ ] Release code for replicate on BARF
- [ ] Release code for replicate on EventNeRF
- [ ] Release code for replicate on NICE-SLAM
- [ ] Make this page look nicer :)

