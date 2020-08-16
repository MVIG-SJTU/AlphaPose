# Getting Started

## Flags
Checkout the [run.md](./run.md) for all flags.

## Example Inference
- **Input dir**: Run AlphaPose for all images in a folder with:
``` bash
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --indir ${img_directory} --outdir ${output_directory}
```
- **Video**:  Run AlphaPose for a video and save the rendered video with:
``` bash
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --video ${path to video} --outdir examples/res --save_video
```
- **Webcam**:  Run AlphaPose using default webcam and visualize the results with:
``` bash
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --outdir examples/res --vis --webcam 0
```
- **Input list**:  Run AlphaPose for images in a list and save the rendered images with:
``` bash
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --list examples/list-coco-demo.txt --indir ${img_directory} --outdir examples/res --save_img
```
- **Only-cpu/Multi-gpus**: Run AlphaPose for images in a list by cpu only or multi gpus:
``` bash
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --list examples/list-coco-demo.txt --indir ${img_directory} --outdir examples/res --gpus ${-1(cpu only)/0,1,2,3(multi-gpus)}
```
- **Re-ID Track(Experimental)**: Run AlphaPose for tracking persons in a video by human re-id algorithm:
``` bash
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --video ${path to video} --outdir examples/res --pose_track --save_video
```
- **Simple Track(Experimental)**: Run AlphaPose for tracking persons in a video by MOT tracking algorithm:
``` bash
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --video ${path to video} --outdir examples/res --detector tracker --save_video
```
- **Pose Flow(not ready)**: Run AlphaPose for tracking persons in a video by embedded PoseFlow algorithm:
``` bash
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --video ${path to video} --outdir examples/res --pose_flow --save_video
```


## Options
- **Note**:  If you meet OOM(out of memory) problem, decreasing the pose estimation batch until the program can run on your computer:
``` bash
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --indir ${img_directory} --outdir examples/res --detbatch 1 --posebatch 30
```
- **Getting more accurate**: You can use larger input for pose network to improve performance e.g.:
```
python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --indir ${img_directory} --outdir ${output_directory} --flip
```
- **Speeding up**:  Checkout the [speed_up.md](./speed_up.md) for more details.

## Output format
Checkout the [output.md](./output.md) for more details.
