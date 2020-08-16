# Pose Tracking Module for AlphaPose

AlphaPose provide three different tracking methods for now, you can try different method to see which one is better for you.

## 1. Human-ReID based tracking (Recommended)
Currently the best performance tracking model. Paper coming soon.

#### Getting started
Download  [human reid model](https://mega.nz/#!YTZFnSJY!wlbo_5oa2TpDAGyWCTKTX1hh4d6DvJhh_RUA2z6i_so) and place it into `AlphaPose/trackers/weights/`.

Then simply run alphapose with additional flag `--pose_track`

You can try different person reid model by modifing `cfg.arch` and `cfg.loadmodel` in `./trackers/tracker_cfg.py`.

If you want to train your own reid model, please refer to this [project](https://github.com/KaiyangZhou/deep-person-reid)

#### Demo
``` bash
./scripts/inference.sh ${CONFIG} ${CHECKPOINT} ${VIDEO_NAME}  ${OUTPUT_DIR}, --pose_track
```
#### Todo
- [] Evaluation Tools for PoseTrack
- [] More Models
- [] Training code for [PoseTrack Dataset](https://posetrack.net/)

## 2. Detector based human tracking
Use a human detecter with tracking module (JDE). Please refer to [detector/tracker/](../detector/tracker/)

#### Getting started
Download detector [JDE-1088x608](https://github.com/Zhongdao/Towards-Realtime-MOT#pretrained-model-and-baseline-models) and place it under `AlphaPose/detector/tracker/data/`

Enable tracking by setting the detector as tracker: `--detector tracker`
#### Demo
``` bash
./scripts/inference.sh ${CONFIG} ${CHECKPOINT} ${VIDEO_NAME}  ${OUTPUT_DIR}, --detector tracker
```

## 3. PoseFlow human tracking
This tracker is based on our BMVC 2018 paper PoseFlow, for more info please refer to [PoseFlow/README.md](PoseFlow/)

#### Getting started

Simply run alphapose with additional flag `--pose_flow`
