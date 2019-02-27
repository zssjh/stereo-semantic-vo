# stereo-semantic-vo

A gif of running on KITTI:

![image](https://github.com/zssjh/stereo-semantic-vo/blob/master/gif/view.gif)

In addition to the requirements in ORB-SLAM2, it also need CUDA>=6.5 to run YOLO detection, 
according to your own GPU compute capability to modify the following line:
```
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
```
My Graphics card is：GeForce GT 730， GPU compute capability=35


