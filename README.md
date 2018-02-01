## Alpha Pose
Alpha Pose is a very accurate multi-person pose estimation system. It is the first open-sourced system that can achieve 70+ mAP (73.2 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset

## Contents
1. [Alpha Pose](#alpha-pose)
2. [Results](#results)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Output](#output)
6. [Speeding Up Alpha Pose](#speeding-up-alpha-pose)
7. [Contributors](#contributors)
9. [Citation](#citation)
10. [License](#license)



## Results
Results on COCO test-dev 2015:
<center>

| Method | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|
| **OpenPose (CMU)** | **61.8** | **83.7** | **69.8** | 69.8 | 69.8 |
| **Detectron (FAIR)** | **61.8** | **83.7** | **69.8** | **69.8** | **69.8** |
| **AlphaPose** | **61.8** | **83.7** | **69.8** | **69.8** | **69.8** |

</center>

Results on MPII full test set:
<center>

| Method | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Ave |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| OpenPose (CMU) | 91.2 | 87.6 | 77.7 | 66.8 | 75.4 | 68.9 | 61.7 | 75.6 |
| Newell & Deng | 92.1 | 89.3 | 78.9 | 69.8 | 76.2 | 71.6 | 64.7 | 77.5 |
| **Alpha Pose** | 92.1 | 89.3 | 78.9 | 69.8 | 76.2 | 71.6 | 64.7 | 82.1 |

</center>

### Pose Estimation

### Pose Tracking


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
2. Install [Torch](https://github.com/torch/distro) and [TensorFlow](https://www.tensorflow.org/install/)(verson >= 1.2).
  ```Shell
  chmod +x install.sh
  ./install.sh
  ```
3. Run fetch_models.sh to download our pre-trained models.
  ```Shell
  chmod +x fetch_models.sh
  ./fetch_models.sh
  ```

## Quick Start
- **Demo**:  Run AlphaPose for all images in a folder and visualize the results with:
```
./run.sh --indir examples/demo/ --outdir examples/results/ --vis
```

The visualized results will be stored in examples/results/RENDER. To easily process images/video and display/save the results, please see [doc/run.md](doc/run.md).

## Output
Output (format, keypoint index ordering, etc.) in [doc/output.md](doc/output.md).



## Speeding Up Alpha Pose


## Contributors


## Citation



## License
AlphaPose is freely available for free non-commercial use, and may be redistributed under these conditions. 
