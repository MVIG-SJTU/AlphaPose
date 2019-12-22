AlphaPose - Speeding Up
============================================


Run AlphaPose for a video, speeding up by increasing the confidence and lowering the NMS threshold:
```
python3 video_demo.py --video ${path to video} --outdir examples/results/  --conf 0.5 --nms 0.45
```
For users with GPU memory >= 8GB, I suggest increasing the detection batch:
```
python3 demo.py --indir ${img_directory} --outdir examples/res --detbatch 2
```
For users that do not need to detect small size persons, I suggest lowering the input size of detection network. The inp_dim should be multiple of 32.
```
python3 demo.py --indir ${img_directory} --outdir examples/res --inp_dim 480
```