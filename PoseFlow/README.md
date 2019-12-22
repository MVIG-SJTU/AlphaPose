# Pose Flow

Official implementation of [Pose Flow: Efficient Online Pose Tracking ](https://arxiv.org/abs/1802.00977).

<p align='center'>
    <img src="posetrack1.gif", width="360">
    <img src="posetrack2.gif", width="344">
</p>

Results on PoseTrack Challenge validation set:

1. Task2: Multi-Person Pose Estimation (mAP)
<center>

| Method | Head mAP | Shoulder mAP | Elbow mAP | Wrist mAP | Hip mAP | Knee mAP | Ankle mAP | Total mAP |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Detect-and-Track(FAIR) | **67.5** | 70.2 | 62 | 51.7 | 60.7 | 58.7 | 49.8 | 60.6 |
| **AlphaPose** | 66.7 | **73.3** | **68.3** | **61.1** | **67.5** | **67.0** | **61.3** | **66.5** |

</center>

2. Task3: Pose Tracking (MOTA)
<center>

| Method | Head MOTA | Shoulder MOTA | Elbow MOTA | Wrist MOTA | Hip MOTA | Knee MOTA | Ankle MOTA | Total MOTA | Total MOTP| Speed(FPS) |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Detect-and-Track(FAIR) | **61.7** | 65.5 | 57.3 | 45.7 | 54.3 | 53.1 | 45.7 | 55.2 | 61.5 |Unknown|
| **PoseFlow(DeepMatch)** | 59.8 | **67.0** | 59.8 | 51.6 | **60.0** | **58.4** | **50.5** | **58.3** | **67.8**|8|
| **PoseFlow(OrbMatch)** | 59.0 | 66.8 | **60.0** | **51.8** | 59.4 | **58.4** | 50.3 | 58.0 | 62.2|24|

</center>

## Latest Features
- Dec 2018: <strong>PoseFlow(General Version)</strong> released! Support ANY DATASET and pose tracking results visualization.
- Oct 2018: Support generating correspondence files with ORB(OpenCV), 3X FASTER and no need to compile DeepMatching library. 

## Requirements

- Python 2.7.13
- OpenCV 3.4.2.16
- OpenCV-contrib 3.4.2.16
- tqdm 4.19.8

## Installation

1. Download PoseTrack Dataset from [PoseTrack](https://posetrack.net/) to `AlphaPose/PoseFlow/posetrack_data/`
2. (Optional) Use [DeepMatching](http://lear.inrialpes.fr/src/deepmatching/) to extract dense correspondences between adjcent frames in every video, please refer to [DeepMatching Compile Error](https://github.com/MVIG-SJTU/AlphaPose/issues/97) to compile DeepMatching correctly

```shell
pip install -r requirements.txt

cd deepmatching
make clean all
make
cd ..
```

## For Any Datasets (General Version)

1. Using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to generate multi-person pose estimation results.

```shell
# pytorch version
python demo.py --indir ${image_dir}$ --outdir ${results_dir}$

# torch version
./run.sh --indir ${image_dir}$ --outdir ${results_dir}$
```

2. Run pose tracking


```shell
# pytorch version
python tracker-general.py --imgdir ${image_dir}$ 
                          --in_json ${results_dir}$/alphapose-results.json 
                          --out_json ${results_dir}$/alphapose-results-forvis-tracked.json
                          --visdir ${render_dir}$

# torch version
python tracker-general.py --imgdir ${image_dir}$ 
                          --in_json ${results_dir}$/POSE/alpha-pose-results-forvis.json 
                          --out_json ${results_dir}$/POSE/alpha-pose-results-forvis-tracked.json
                          --visdir ${render_dir}$
```


## For PoseTrack Dataset Evaluation (Paper Baseline)

1. Using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to generate multi-person pose estimation results on videos with format like `alpha-pose-results-sample.json`.
2. Using DeepMatching/ORB to generate correspondence files.

```shell
# Generate correspondences by DeepMatching
# (More Robust but Slower)
python matching.py --orb=0 

or

# Generate correspondences by Orb
# (Faster but Less Robust)
python matching.py --orb=1
```

3. Run pose tracking


```shell
python tracker-baseline.py --dataset=val/test  --orb=1/0
```
4. Evaluation

Original [poseval](https://github.com/leonid-pishchulin/poseval) has some instructions on how to convert annotation files from MAT to JSON.

Evaluate pose tracking results on validation dataset:

```shell
git clone https://github.com/leonid-pishchulin/poseval.git --recursive
cd poseval/py && export PYTHONPATH=$PWD/../py-motmetrics:$PYTHONPATH
cd ../../
python poseval/py/evaluate.py --groundTruth=./posetrack_data/annotations/val \
                    --predictions=./${track_result_dir}/ \
                    --evalPoseTracking --evalPoseEstimation
```


## Citation

Please cite these papers in your publications if it helps your research:

    @inproceedings{xiu2018poseflow,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      booktitle={BMVC},
      year = {2018}
    }





