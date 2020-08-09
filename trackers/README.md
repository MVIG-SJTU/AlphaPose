# Pose Tracking
## Models
Download  [human reid model](https://mega.nz/#!YTZFnSJY!wlbo_5oa2TpDAGyWCTKTX1hh4d6DvJhh_RUA2z6i_so) and place it into `./trackers/weights/`.

You can try different person reid model by modifing `cfg.arch` and `cfg.loadmodel` in `./trackers/tracker_cfg.py`.

If you want to train your own reid model, please refer to this [project](https://github.com/KaiyangZhou/deep-person-reid)
## Demo
``` bash
./scripts/inference.sh ${CONFIG} ${CHECKPOINT} ${VIDEO_NAME}  ${OUTPUT_DIR}, --pose_track
```
## Todo
- [] Evaluation Tools for PoseTrack
- [] More Models
- [] Training code for [PoseTrack Dataset](https://posetrack.net/)



