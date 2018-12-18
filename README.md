
<div align="center">
    <img src="doc/logo.jpg", width="400">
</div>



## AlphaPose
[Alpha Pose](http://www.mvig.org/research/alphapose.html) is an accurate multi-person pose estimator, which is the **first real-time** open-source system that achieves **70+ mAP (72.3 mAP)** on COCO dataset and **80+ mAP (82.1 mAP)** on MPII dataset.** 
To match poses that correspond to the same person across frames, we also provide an efficient online pose tracker called Pose Flow. It is the **first open-source online pose tracker that achieves both 60+ mAP (66.5 mAP) and 50+ MOTA (58.3 MOTA) on PoseTrack Challenge dataset.**


## News!
- Dec 2018: [General version](https://github.com/MVIG-SJTU/AlphaPose/tree/master/PoseFlow) of PoseFlow is released! 3X Faster and support pose tracking results visualization!
- Sep 2018: [**PyTorch** version](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch) of AlphaPose is released! It runs at **20 fps** on COCO validation set (4.6 people per image on average) and achieves 71 mAP!

## Contents
- [AlphaPose](#alphapose)
- [News!](#news)
- [Contents](#contents)
- [Results](#results)
  - [Pose Estimation](#pose-estimation)
  - [Pose Tracking](#pose-tracking)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Output](#output)
- [Speeding Up AlphaPose](#speeding-up-alphapose)
- [Feedbacks](#feedbacks)
- [Contributors](#contributors)
- [Citation](#citation)
- [License](#license)



## Results
### Pose Estimation
<p align="center">
    <img src="doc/pose.gif", width="360">
</p>

Results on COCO test-dev 2015:
<center>

| Method | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|
| OpenPose (CMU-Pose) | 61.8 | 84.9 | 67.5 | 57.1 | 68.2 |
| Detectron (Mask R-CNN) | 67.0 | 88.0 | 73.1 | 62.2 | 75.6 |
| **AlphaPose** | **72.3** | **89.2** | **79.1** | **69.0** | **78.6** |

</center>

Results on MPII full test set:
<center>

| Method | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Ave |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| OpenPose (CMU-Pose) | 91.2 | 87.6 | 77.7 | 66.8 | 75.4 | 68.9 | 61.7 | 75.6 |
| Newell & Deng | **92.1** | 89.3 | 78.9 | 69.8 | 76.2 | 71.6 | 64.7 | 77.5 |
| **AlphaPose** | 91.3 | **90.5** | **84.0** | **76.4** | **80.3** | **79.9** | **72.4** | **82.1** |

</center>

### Pose Tracking
<p align='center'>
    <img src="doc/posetrack.gif", width="360">
    <img src="doc/posetrack2.gif", width="344">
</p>

Results on PoseTrack Challenge validation set:

1. Task2: Multi-Person Pose Estimation (mAP)
<center>

| Method | Head mAP | Shoulder mAP | Elbow mAP | Wrist mAP | Hip mAP | Knee mAP | Ankle mAP | Total mAP |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Detect-and-Track(FAIR) | **67.5** | 70.2 | 62 | 51.7 | 60.7 | 58.7 | 49.8 | 60.6 |
| **AlphaPose** | 66.7 | **73.3** | **68.3** | **61.1** | **67.5** | **67.0** | **61.3** | **66.5** |

</center>

2. Task3: Pose Tracking (MOTA)
<center>

| Method | Head MOTA | Shoulder MOTA | Elbow MOTA | Wrist MOTA | Hip MOTA | Knee MOTA | Ankle MOTA | Total MOTA | Total MOTP| Speed(FPS) |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Detect-and-Track(FAIR) | **61.7** | 65.5 | 57.3 | 45.7 | 54.3 | 53.1 | 45.7 | 55.2 | 61.5 |Unknown|
| **PoseFlow(DeepMatch)** | 59.8 | **67.0** | 59.8 | 51.6 | **60.0** | **58.4** | **50.5** | **58.3** | **67.8**|8|
| **PoseFlow(OrbMatch)** | 59.0 | 66.8 | **60.0** | **51.8** | 59.4 | **58.4** | 50.3 | 58.0 | 62.2|24|

</center>

*Note: Please read [PoseFlow/README.md](PoseFlow/) for details.*

## Installation
1. Get the code and build related modules.
  ```Shell
  git clone https://github.com/MVIG-SJTU/AlphaPose.git
  cd AlphaPose/human-detection/lib/
  make clean
  make
  cd newnms/
  make
  cd ../../../
  ```
2. Install [Torch](https://github.com/torch/distro) and [TensorFlow](https://www.tensorflow.org/install/)(verson >= 1.2). After that, install related dependencies by:
  ```Shell
  chmod +x install.sh
  ./install.sh
  ```
3. Run fetch_models.sh to download our pre-trained models. Or download the models manually: output.zip([Google drive](https://drive.google.com/open?id=1dMiUPMvt5o-S1BjDkzUJooEoT3GgasxB)|[Baidu pan](https://pan.baidu.com/s/1hund0US)), final_model.t7([Google drive](https://drive.google.com/open?id=1JYlLspGJHJFIggkDll4MdUdqX2ELqHpk)|[Baidu pan](https://pan.baidu.com/s/1qZuEyF6))
  ```Shell
  chmod +x fetch_models.sh
  ./fetch_models.sh
  ```

## Quick Start
- **Demo**:  Run AlphaPose for all images in a folder and visualize the results with:
```
./run.sh --indir examples/demo/ --outdir examples/results/ --vis
```

The visualized results will be stored in examples/results/RENDER. To easily process images/video and display/save the results, please see [doc/run.md](doc/run.md). **If you get any problems, you can check the [doc/faq.md](doc/faq.md).**

- **Video**:  You can see our video demo [here](https://www.youtube.com/watch?v=Z2WPd59pRi8).


## Output
Output (format, keypoint index ordering, etc.) in [doc/output.md](doc/output.md).

## Speeding Up AlphaPose
We provide a `fast` mode for human-detection that disables multi-scale testing. You can turn it on by adding `--mode fast`.

And if you have multiple gpus on your machine or have large gpu memories, you can speed up the pose estimation step by using multi-gpu testing or large batch tesing with:
```
./run.sh --indir examples/demo/ --outdir examples/results/ --gpu 0,1,2,3 --batch 5
```
It assumes that you have 4 gpu cards on your machine and *each card* can run a batch of 5 images. Here is the recommended batch size for gpu with different size of memory:
```
GPU memory: 4GB -- batch size: 3
GPU memory: 8GB -- batch size: 6
GPU memory: 12GB -- batch size: 9
```
See [doc/run.md](doc/run.md) for more details.


## Feedbacks
If you get any problems, you can check the [doc/faq.md](doc/faq.md) first. If it can not solve your problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request!


## Contributors
AlphaPose is based on RMPE(ICCV'17), authored by [Hao-shu Fang](https://fang-haoshu.github.io/), Shuqin Xie, [Yu-Wing Tai](https://scholar.google.com/citations?user=nFhLmFkAAAAJ&hl=en) and [Cewu Lu](http://www.mvig.org/), [Cewu Lu](http://mvig.sjtu.edu.cn/) is the corresponding author. Currently, it is developed and maintained by [Hao-shu Fang](https://fang-haoshu.github.io/), [Jiefeng Li](http://jeff-leaf.site/), [Yuliang Xiu](http://xiuyuliang.cn/about/) and [Ruiheng Chang](https://crh19970307.github.io/). 

The main contributors are listed in [doc/contributors.md](doc/contributors.md).

## Citation
Please cite these papers in your publications if it helps your research:

    @inproceedings{fang2017rmpe,
      title={{RMPE}: Regional Multi-person Pose Estimation},
      author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
      booktitle={ICCV},
      year={2017}
    }

    @inproceedings{xiu2018poseflow,
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      booktitle={BMVC},
      year = {2018}
    }


## License
AlphaPose is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, contact [Cewu Lu](http://www.mvig.org/)
