# Pose Tracking Module for AlphaPose


## Human-ReID based tracking (Recommended)
Currently the best performance tracking model. Paper coming soon.

### Getting started
Download  [human reid model](https://mega.nz/#!YTZFnSJY!wlbo_5oa2TpDAGyWCTKTX1hh4d6DvJhh_RUA2z6i_so) and place it into `./trackers/weights/`.

Then simply run alphapose with additional flag `--pose_track`

You can try different person reid model by modifing `cfg.arch` and `cfg.loadmodel` in `./trackers/tracker_cfg.py`.

If you want to train your own reid model, please refer to this [project](https://github.com/KaiyangZhou/deep-person-reid)

### Demo
``` bash
./scripts/inference.sh ${CONFIG} ${CHECKPOINT} ${VIDEO_NAME}  ${OUTPUT_DIR}, --pose_track
```
### Todo
- [] Evaluation Tools for PoseTrack
- [] More Models
- [] Training code for [PoseTrack Dataset](https://posetrack.net/)


## PoseFlow human tracking
This tracker is based on our BMVC 2018 paper PoseFlow.

### Getting started

Simply run alphapose with additional flag `--pose_flow`

### More info
For more info, please refer to [PoseFlow/README.md](PoseFlow/)