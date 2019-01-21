The repository contains two ways of adding semantic information to visual odometry

***Online semantic***
The folder /coco, /kitti contain the names,cfg and weights of coco and kitti dataset, respectively, which are used to detect objects
as a parallel thread with visual odometry. Because there is only 2G memory on my graphics card, so I can only drive yolov2-tiny online,
whose performance is Just passable when the detection threshold is set to 0.8.


***Offline semantic***
The folders /*results contain offline detection results of yolov3 on NVIDIAGTX1080.We read the results directly.

