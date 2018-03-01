# Pose Flow

Official implementation of [Pose Flow: Efficient Online Pose Tracking ](https://arxiv.org/abs/1802.00977).

## Installation

1. Download PoseTrack Dataset from [PoseTrack](https://posetrack.net/) to `AlphaPose/poseflow/posetrack_data/`
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

Run pose tracking (python3)
```shell
python tracker.py
```

## Evaluation

Evaluate pose tracking results on validation dataset (python2)
```shell
cd poseval/py && export PYTHONPATH=$PWD/../py-motmetrics:$PYTHONPATH
python evaluate.py --groundTruth=/AlphaPose/poseflow/posetrack_data/annotations/val \
                    --predictions=/AlphaPose/poseflow/${track_dir}/ \
                    --evalPoseTracking --evalPoseEstimation
```
## Citation
Please cite these papers in your publications if it helps your research:

    @ARTICLE{2018arXiv180200977X,
      author = {{Xiu}, Y. and {Li}, J. and {Wang}, H. and {Fang}, Y. and {Lu}, C.},
      title = "{Pose Flow: Efficient Online Pose Tracking}",
      journal = {ArXiv e-prints},
      eprint = {1802.00977},
      year = 2018
    }



