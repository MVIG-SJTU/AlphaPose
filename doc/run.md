Alpha Pose run.sh Usage & Examples
====================================

We provide a script `run.sh` to ease your effort of running our code. Here, we first the flags of this script and then give some examples.

## Flags
- `--gpu`: Which GPU(s) to use. 
- `--batch`: Batch size for the pose estimation network.
- `--indir`: Directory of the input images. All the images in the directory will be processed.
- `--list`: A txt list for the input images, each line is the absolute path to the image. Not co-appear with `--indir`
- `--video`: Read video and process the video frame by frame.
- `--outdir`: Output directory to store the human detection and pose estimation results.
- `--mode`: fast/normal/accurate. We recommend using the mode 'normal'
- `--vis`: If turned-on, it will visualize the results and save them as images. If the input is video, it will save the output as video (but without audio).
- `--seperate-json`: If turned on, it will save the json file for each image/frame of the input. Default is false.
- `--format`: The format of the saved results. By default, it will save the output in COCO-like format. Option is 'cmu', which save the results in the format of CMU-Pose. For more details, see [format.md](format.md)

## Examples
