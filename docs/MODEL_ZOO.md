# MODEL ZOO

## MSCOCO dataset 

| Model                    | Backbone | Detector | Input Size |     AP     | Speed |  Download | Config | Training Log |  
|--------------------------|----------|----------|------------|------------|-------|-----------|--------|--------------|
|[Simple Baseline](../configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml)    | ResNet50           | YOLOv3 | 256x192            | 70.6        | 2.94 iter/s | [model](https://drive.google.com/open?id=1nxyfUbvWDaaT9eDM7Y31ScSVPlGy6gfw) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml)    | [log](logs/simple_res50_256x192.log) |
|[Fast Pose](../configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml)    | ResNet50           | YOLOv3 | 256x192            | 72.0        | 3.54 iter/s | [model](https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml)    | [log](logs/fast_res50_256x192.log) |
|[Fast Pose (DUC)](../configs/coco/resnet/256x192_res50_lr1e-3_1x-duc.yaml)    | ResNet50 - unshuffle           | YOLOv3 | 256x192            | 72.4        | 2.91 iter/s | [model](https://drive.google.com/open?id=1RlnL_YnnmgOM4L9L2szFdUnTjwptqtL-) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_1x-duc.yaml)    | [log](logs/fast_421_res50-shuffle_256x192.log) |
|[HRNet](../configs/coco/hrnet/256x192_w32_lr1e-3.yaml)    | HRNet-W32           | YOLOv3 | 256x192            | 72.5        | 2.13 iter/s | [model](https://drive.google.com/open?id=1i63BPlOnp2vSjIZ7ni4Yp3RCPQwqe922) | [cfg](../configs/coco/hrnet/256x192_w32_lr1e-3.yaml)    | [log](logs/hrnet_w32_256x192.log) |
|[Fast Pose (DCN)](../configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | 72.8        | 2.94 iter/s | [model](https://drive.google.com/open?id=1zUz9YIk6eALCbZrukxD7kQ554nhi1pVv) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml)    | [log](logs/fast_dcn_res50_256x192.log) |
|[Fast Pose (DUC)](../configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    | ResNet152           | YOLOv3 | 256x192            | 73.3        | 1.62 iter/s | [model](https://drive.google.com/open?id=1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9) | [cfg](../configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    | [log](logs/fast_421_res152_256x192.log) |

#### Notes
- All models are trained on keypoint train 2017 images which contains at least one human with keypoint annotations (64115 images).
- The evaluation is done on COCO keypoint val 2017 (5000 images).
- Flip test is used by default.
- One TITAN XP is used for speed test, with `batch_size=64` in each iteration.
- Offline human detection results are used in speed test.
- `FastPose` is our own network design. Paper coming soon!


## [Halpe dataset](https://github.com/Fang-Haoshu/Halpe-FullBody) (26 keypoints)

| Model                    | Backbone | Detector | Input Size |     AP     | Speed |  Download | Config |  
|--------------------------|----------|----------|------------|------------|-------|-----------|--------|
|[Fast Pose](../configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml)    | ResNet50           | YOLOv3 | 256x192            | -        | 13.12 iter/s | [Google](https://drive.google.com/file/d/1S-ROA28de-1zvLv-hVfPFJ5tFBYOSITb/view?usp=sharing) [Baidu](https://pan.baidu.com/s/1lvzMhoYgS6o6n8lVDx3GtQ) | [cfg](../configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml)    |

For example, you can run with:
```
python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir examples/demo/ --save_img
```

#### Notes
- This model is trained based on the first 26 keypoints of Halpe Full-body datatset (without face and hand keypoints).
- The speed is tested on COCO val2017 on a single NVIDIA GeForce RTX 3090 gpu, with `batch_size=64` in each iteration and offline yolov3 human detection results.

## [Halpe dataset](https://github.com/Fang-Haoshu/Halpe-FullBody) (136 keypoints)

| Model                    | Backbone | Detector | Input Size | Loss Type |     AP     | Speed |  Download | Config |  
|--------------------------|----------|----------|------------|------------|------------|-------|-----------|--------|
|[Fast Pose](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x.yaml)    | ResNet50           | YOLOv3 | 256x192            | Heatmap | 41.7        | 4.37 iter/s | [Google](halpe136_fast50_256x192.pth) [Baidu](halpe136_fast50_256x192.pth) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    |
|[Fast Pose](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    | ResNet50           | YOLOv3 | 256x192            | Symmetric Integral | 44.1        | 16.50 iter/s | [Google](halpe136_fast50_regression_256x192.pth) [Baidu](halpe136_fast50_regression_256x192.pth) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    |
|[Fast Pose (DCN)](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Symmetric Integral | 46.2        | 16.58 iter/s | [Google](halpe136_fast50_dcn_regression_256x192.pth) [Baidu](halpe136_fast50_dcn_regression_256x192.pth) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml)    |
|[Fast Pose (DCN)](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined | 45.4        | 10.07 iter/s | [Google](halpe136_fast50_dcn_combined_256x192.pth) [Baidu](halpe136_fast50_dcn_combined_256x192.pth) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    |
|[Fast Pose (DCN)](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined (10 hand weight) | 47.2        | 10.07 iter/s | [Google](halpe136_fast50_dcn_combined_256x192_10handweight.pth) [Baidu](halpe136_fast50_dcn_combined_256x192_10handweight.pth) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    |
|[Fast Pose (DUC)](../configs/halpe_136/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    | ResNet152           | YOLOv3 | 256x192            | Symmetric Integral | 45.1        | 16.17 iter/s | [Google](halpe136_fast152_duc_regression_256x192.pth) [Baidu](halpe136_fast152_duc_regression_256x192.pth) | [cfg](../configs/halpe_136/256x192_res152_lr1e-3_1x-duc.yaml)    |

For example, you can run with:
```
python scripts/demo_inference.py --cfg configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/halpe136_fast50_regression_256x192.pth --indir examples/demo/ --save_img
```

#### Notes
- All of above models are trained only on Halpe Full-body dataset.
- The APs are tested under Halpe's criterion, with flip test on and without NMS.
- Combined loss means we use heatmap loss (mse loss) on body and foot keypoints and use symmetric integral loss (l1 joint regression loss) on face and hand keypoints.
- There are two FastPose-DCN models with combined loss. The second one uses ten times of weight of hand keypoints, so it is more accurate on hand keypoints but less accurate on the other keypoints.
- The speed is tested on COCO val2017 on a single NVIDIA GeForce RTX 3090 gpu, with `batch_size=64` in each iteration and offline yolov3 human detection results.

## [COCO WholeBody dataset](https://github.com/jin-s13/COCO-WholeBody) (133 keypoints)

| Model                    | Backbone | Detector | Input Size | Loss Type |     AP     | Speed |  Download | Config |  
|--------------------------|----------|----------|------------|------------|------------|-------|-----------|--------|
|[Fast Pose](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    | ResNet50           | YOLOv3 | 256x192            | Symmetric Integral | 55.4        | 17.42 iter/s | [Google](coco_wholebody133_fast50_regression_256x192.pth) [Baidu](coco_wholebody133_fast50_regression_256x192.pth) | [cfg](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    |
|[Fast Pose (DCN)](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Symmetric Integral | 57.7        | 16.70 iter/s | [Google](coco_wholebody133_fast50_dcn_regression_256x192.pth) [Baidu](coco_wholebody133_fast50_dcn_regression_256x192.pth) | [cfg](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-combined.yaml)    |
|[Fast Pose](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-combined.yaml)    | ResNet50           | YOLOv3 | 256x192            | Combined | 57.8        | 10.28 iter/s | [Google](coco_wholebody133_fast50_combined_256x192.pth) [Baidu](coco_wholebody133_fast50_combined_256x192.pth) | [cfg](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-combined.yaml)    |
|[Fast Pose (DCN)](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined | 58.2        | 10.22 iter/s | [Google](coco_wholebody133_fast50_dcn_combined_256x192.pth) [Baidu](coco_wholebody133_fast50_dcn_combined_256x192.pth) | [cfg](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    |
|[Fast Pose (DUC)](../configs/coco_wholebody/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    | ResNet152           | YOLOv3 | 256x192            | Symmetric Integral | 56.9        | 15.72 iter/s | [Google](coco_wholebody133_fast152_duc_regression_256x192.pth) [Baidu](coco_wholebody133_fast152_duc_regression_256x192.pth) | [cfg](../configs/coco_wholebody/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    |

#### Notes
- All of above models are trained only on COCO WholeBody dataset.
- The APs are tested under COCO WholeBody's criterion, with flip test on and without NMS.
- The speed is tested on COCO val2017 on a single NVIDIA GeForce RTX 3090 gpu, with `batch_size=64` in each iteration and offline yolov3 human detection results.

## Multi Domain Models (Strongly Recommended)
| Model                    | Backbone | Detector | Input Size | Loss Type |     AP     | Speed |  Download | Config | #keypoints |  
|--------------------------|----------|----------|------------|------------|------------|-------|-----------|--------|--------------|
|[Fast Pose](../configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    | ResNet50           | YOLOv3 | 256x192            | Symmetric Integral | 50.3        | 16.28 iter/s | [Google](multi_domain_fast50_regression_256x192.pth) [Baidu](multi_domain_fast50_regression_256x192.pth) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    | 136 |
|[Fast Pose (DCN)](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined (10 hand weight) | 49.8        | 10.35 iter/s | [Google](multi_domain_fast50_dcn_combined_256x192.pth) [Baidu](multi_domain_fast50_dcn_combined_256x192.pth) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | 136 |
|[Fast Pose (DCN)](../configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined | -        | 13.88 iter/s | [Google](noface_fast50_dcn_combined_256x192.pth) [Baidu](noface_fast50_dcn_combined_256x192.pth) | [cfg](../configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | 68 (no face) |
|[Fast Pose (DCN)](../configs/single_hand/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml)    | ResNet50 - dcn           | - | 256x192            | Symmetric Integral | -        | 30.20 iter/s | [Google](singlehand_fast50_dcn_regression_256x192.pth) [Baidu](singlehand_fast50_dcn_regression_256x192.pth) | [cfg](../configs/single_hand/resnet/256x192_res50_lr1e-3_2x-dcn_regression.yaml)    | 21 (single hand) |

For the most accurate wholebody pose estimation, you can run with:
```
python scripts/demo_inference.py --cfg configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml --checkpoint pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth --indir examples/demo/ --save_img
```
or, you can run with (this version is a little faster and more accurate on body keypoints, but its performance on hand keypoints are worser):
```
python scripts/demo_inference.py --cfg configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/multi_domain_fast50_regression_256x192.pth --indir examples/demo/ --save_img
```

#### Notes
- These models are strongly recommended because they are more accurate and flexible.
- These models are trained with multi-domain knowledge distillation (MDKD, see our [paper]() for more details).
- The APs are tested under Halpe's criterion, with flip test on and without NMS.
- If you want to use the single hand model, you should give the rough bounding box of a single hand instead of that of a whole person.
- The speed is tested on COCO val2017 on a single NVIDIA GeForce RTX 3090 gpu, with `batch_size=64` in each iteration and offline yolov3 human detection results.
