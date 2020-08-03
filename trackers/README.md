# Pose Tracking
## Models
Download our [human reid model](https://jbox.sjtu.edu.cn/l/I51LcR) and place it into `./trackers`.

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



