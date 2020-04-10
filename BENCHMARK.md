# Benchmark on AlphaPose architecture

## 1. Using different detect network

使用yolo和centerpose作为检测框的对比：

pose arch     | groundtruth | yolo    | centerpose
--------------|-------------|---------|--------------
resnet50      | 0.732       | 0.712   | 0.646

## 2. Using different SPPE network

使用yolo和centerpose作为检测框，使用resnet50/resnet18/resnet18_aug作为SPPE框架：

pose arch     | groundtruth | yolo    | centerpose
--------------|-------------|---------|--------------
resnet50      | 0.732       | 0.712   | 0.646
resnet18      | -           | -       | - 
resent18_aug  | -           | -       | -

