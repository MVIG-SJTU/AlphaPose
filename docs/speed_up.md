AlphaPose - Speeding Up
============================================


1. Run AlphaPose for a video, speeding up by increasing the confidence, lowering the NMS threshold, lowering the input resolution of detector in `detector/yolo_cfg.py`
```
cfg.NMS_THRES =  0.45
cfg.CONFIDENCE = 0.5
cfg.INP_DIM =  420
```
It may miss some people though.

2. Increase the detbatch and posebatch by setting the `--detbatch` and `--posebatch` flag if you have large GPU memory.
