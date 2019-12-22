# faster-rcnn
A Tensorflow implementation of faster RCNN detection framework with soft-nms. Modified from [Xinlei Chen's implementation](https://github.com/endernewton/tf-faster-rcnn).

### Detection Performance
We use ResNet152 as our backbone. Combined with multi-scale testing and soft-nms, the detection accuracy on COCO test-dev set is around **40.0** mAP. We do not finetune the detector on person category since we found that it made no differences on the final pose estimation performance.

### Usage
```
python demo-alpha-pose.py --inputlist=${LIST_FILE} --inputpath=${INPUT_PATH} --outputpath=${OUTPUT_PATH} --mode=${MODE}
  --inputlist: A text file list for the input images, each line is the **ASOLUTE PATH** to the image. Not co-appear with `--indir`
  --inputpath:  Directory of the input images. All the images in the directory will be processed.
  --outputpath: Output directory to store the human detection and pose estimation results.
  --mode: If set as 'fast', then disable multi-scale testing
```
