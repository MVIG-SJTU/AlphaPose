AlphaPose Usage & Examples
====================================

Here, we first list the flags of this script and then give some examples.

## Flags
- `--indir`: Directory of the input images. All the images in the directory will be processed.
- `--list`: A text file list for the input images
- `--video`: Read video and process the video frame by frame.
- `--outdir`: Output directory to store the pose estimation results.
- `--vis`: If turned-on, it will render the results and visualize them.
- `--save_img`: If turned-on, it will render the results and save them as images in $outdir/vis. 
- `--save_video`: If turned-on, it will render the results and save them as a video.
- `--vis_fast`: If turned on, it will use faster rendering method. Default is false.
- `--format`: The format of the saved results. By default, it will save the output in COCO-like format. Alternative options are 'cmu' and 'open', which saves the results in the format of CMU-Pose or OpenPose. For more details, see [output.md](output.md)
- `--conf`: Confidence threshold for human detection. Lower the value can improve the final accuracy but decrease the speed. Default is 0.1.
- `--nms`: Confidence threshold for human detection. Increase the value can improve the final accuracy but decrease the speed. Default is 0.6.
- `--detbatch`: Batch size for the detection network. 
- `--posebatch`: Maximum batch size for the pose estimation network. If you met OOM problem, decrease this value until it fit in the memory.
- `--sp`: Run the program using a single process. Windows users need to turn this flag on.
- `--inp_dim`: The input size of detection network. The inp_dim should be multiple of 32. Default is 608.

## Examples
- **Run AlphaPose for all images in a folder ,save the results in the format of CMU-Pose and save the images**:
```
python3 demo.py --indir examples/demo/ --outdir examples/results/ --save_img --format cmu
```
- **Run AlphaPose for a video, save the video and use faster rendering method**:
```
python3 video_demo.py --video examples/input.mp4 --outdir examples/results/ --save_video  --vis_fast
```
- **Run AlphaPose for a video, speeding up by increasing the confidence and lowering the NMS threshold.**:
```
python3 video_demo.py --video examples/input.mp4 --outdir examples/results/  --conf 0.5 --nms 0.45
```
