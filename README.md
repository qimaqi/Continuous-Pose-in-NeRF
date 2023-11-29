# Continuous-Pose-in-NeRF
Implementation of Continuous Pose for Monocular Cameras in Neural Implicit Representation


## TODO
- [x] Upload demo code
- [ ] Release pip package (XD new user suspend)
- [ ] Release example on Nerfstudio
- [ ] Release mono-lego dataset
- [ ] Release code for replicate on BARF
- [ ] Release code for replicate on EventNeRF
- [ ] Release code for replicate on NICE-SLAM
- [ ] Make this page look nicer :)


## Quick start
Regarding the environment you need nothing but numpy, torch to make it work
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
That's it
