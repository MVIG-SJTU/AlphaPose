AlphaPose Usage & Examples
====================================

Here, we first list the flags and other parameters you can tune. Default parameters work well and you don't need to tune them unless you know what you are doing.

## Flags
- `--cfg`: Experiment configure file name
- `--checkpoint`: Experiment checkpoint file name
- `--sp`: Run the program using a single process. Windows users need to turn this flag on.
- `--detector`: Detector you can use, yolo/tracker.
- `--indir`: Directory of the input images. All the images in the directory will be processed.
- `--list`: A text file list for the input images
- `--image`: Read single image and process.
- `--video`: Read video and process the video frame by frame.
- `--outdir`: Output directory to store the pose estimation results.
- `--vis`: If turned-on, it will render the results and visualize them.
- `--save_img`: If turned-on, it will render the results and save them as images in $outdir/vis. 
- `--save_video`: If turned-on, it will render the results and save them as a video.
- `--vis_fast`: If turned on, it will use faster rendering method. Default is false.
- `--format`: The format of the saved results. By default, it will save the output in COCO-like format. Alternative options are 'cmu' and 'open', which saves the results in the format of CMU-Pose or OpenPose. For more details, see [output.md](output.md)

- `--detbatch`: Batch size for the detection network. 
- `--posebatch`: Maximum batch size for the pose estimation network. If you met OOM problem, decrease this value until it fit in the memory.
- `--flip`: Enable flip testing. Can increase the accuracy.
- `--min_box_area`: Min box area to filter out, you can set it like 100 to filter out small people.
- `--gpus`: Choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)

- `--pose_track`: Enable tracking pipeline with human re-id feature, it is currently the best performance pose tracker
- `--pose_flow`: This flag will be depreciated. It enables the old tracking version of PoseFlow.

All the flags available here: [link](../scripts/demo_inference.py#L22)


## Parameters
1. yolo detector config is [here](../detector/yolo_cfg.py)
- `CONFIDENCE`: Confidence threshold for human detection. Lower the value can improve the final accuracy but decrease the speed. Default is 0.05.
- `NMS_THRES`: NMS threshold for human detection. Increase the value can improve the final accuracy but decrease the speed. Default is 0.6.
- `INP_DIM`: The input size of detection network. The inp_dim should be multiple of 32. Default is 608. Increase it may improve the accuracy.
