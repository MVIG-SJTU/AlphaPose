# Pose Flow

Official implementation of [Pose Flow: Efficient Online Pose Tracking ](https://arxiv.org/abs/1802.00977).

## Requirements:

- Python 2.7.13

## Installation

1. Download PoseTrack Dataset from [PoseTrack](https://posetrack.net/) to `AlphaPose/PoseFlow/posetrack_data/`
2. Use [DeepMatching](http://lear.inrialpes.fr/src/deepmatching/) to extract dense correspondences between adjcent frames in every video

```shell
pip install -r requirements.txt
cd deepmatching
make clean all
make
cd ..
python deepmatching.py
```
## Quick Start

Run pose tracking
```shell
python tracker.py --dataset=val/test
```
## Evaluation

Original [poseval](https://github.com/leonid-pishchulin/poseval) has some instructions on how to convert annotation files from MAT to JSON.

Evaluate pose tracking results on validation dataset:

```shell
git clone https://github.com/leonid-pishchulin/poseval.git --recursive
cd poseval/py && export PYTHONPATH=$PWD/../py-motmetrics:$PYTHONPATH
cd ../../
python poseval/py/evaluate.py --groundTruth=./posetrack_data/annotations/val \
                    --predictions=./${track_result_dir}/ \
                    --evalPoseTracking --evalPoseEstimation
```
## Citation

Please cite these papers in your publications if it helps your research:

    @ARTICLE{2018arXiv180200977X,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      journal = {ArXiv e-prints},
      eprint = {1802.00977},
      year = {2018}
    }



