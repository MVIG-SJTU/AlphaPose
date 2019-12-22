AlphaPose run.sh Usage & Examples
====================================

We provide a script `run.sh` to ease your effort of running our code. Here, we first list the flags of this script and then give some examples.

## Flags
- `--gpu`: Which GPU(s) to use. 
- `--batch`: Batch size of the pose estimation network for each GPU card. 
- `--indir`: Directory of the input images. All the images in the directory will be processed. Input dir should be **RELATIVE PATH** to AlphaPose directory.
- `--list`: A text file list for the input images, each line is the **ASOLUTE PATH** to the image. Not co-appear with `--indir`
- `--video`: Read video and process the video frame by frame.
- `--outdir`: Output directory to store the human detection and pose estimation results.
- `--mode`: fast/normal/accurate. We recommend using the mode 'normal'. Their differences are listed below.

<center>

| MODE | multi-scale human detection | 4 crop pose estimation | accuracy | speed |
|:-------|:-----:|:-------:|:-------:|:-------:|
| fast | no | no | 70.6 | 0.9x |
| normal | yes | no | 71.6 | 1x |
| accurate | yes | yes | 72.3 | 4.9x |

</center>

- `--vis`: If turned-on, it will visualize the results and save them as images. If the input is video, it will save the output as video (but without audio).
- `--sep`: If turned on, it will save the json file for each image/frame of the input. Default is false.
- `--dataset`: Follow the keypoints definition as COCO or MPII dataset. Default is 'MPII', Alternative option is 'COCO'.
- `--format`: The format of the saved results. By default, it will save the output in COCO-like format. An alternative option is 'cmu', which saves the results in the format of CMU-Pose. For more details, see [format.md](format.md)

## Examples
- **Run AlphaPose for all images in a folder and display the results**:
```
./run.sh --indir examples/demo/ --outdir examples/results/ --vis
```
- **Run AlphaPose for images on a list and save the results in COCO dataset's keypoints order**:
```
./run.sh --list examples/img-list.txt --outdir examples/results/ --dataset COCO
```
- **Run AlphaPose for a video, save the results in CMU-Pose's format and display the results**:
```
./run.sh --video examples/input.mp4 --outdir examples/results/ --vis --format cmu
```
- **Run AlphaPose for all images in 'fast' mode and save the result for each image seperately**:
```
./run.sh --indir examples/demo/ --outdir examples/results/ --mode fast --sep
```
- **Speed up AlphaPose by using multi-gpu and larger batch size. Assumes that you have 2 GPU cards, each card has a memory of 8GB.**:
```
./run.sh --indir examples/demo/ --outdir examples/results/ --gpu 0,1 --batch 6
```
