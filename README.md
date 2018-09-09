
<div align="center">
    <img src="doc/logo.jpg", width="400">
</div>


## News!

This **pytorch** version of AlphaPose runs at **20 fps** on COCO validation set (4.6 people per image on average) and achieves 71 AP!

## AlphaPose
[Alpha Pose](http://www.mvig.org/research/alphapose.html) is an accurate multi-person pose estimator, which is the **first open-source system that achieves 70+ mAP (72.3 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset.** 
To match poses that correspond to the same person across frames, we also provide an efficient online pose tracker called Pose Flow. It is the **first open-source online pose tracker that achieves both 60+ mAP (66.5 mAP) and 50+ MOTA (58.3 MOTA) on PoseTrack Challenge dataset.**

AlphaPose supports both Linux and **Windows!**

<div align="center">
    <img src="doc/alphapose.gif", width="400">
</div>


## Installation
**Windows Version** please check out [doc/win_install.md](doc/win_install.md)

1. Get the code.
  ```Shell
  git clone -b pytorch https://github.com/MVIG-SJTU/AlphaPose.git
  ```

2. Install [pytorch 0.4.0](https://github.com/pytorch/pytorch)
  ```Shell
  chmod +x install.sh
  ./install.sh
  ```

3. Download the models manually: **duc_se.pth** (2018/08/30) ([Google Drive]( https://drive.google.com/open?id=1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW) | [Baidu pan](https://pan.baidu.com/s/15jbRNKuslzm5wRSgUVytrA)), **yolov3.weights**([Google Drive](https://drive.google.com/open?id=1yjrziA2RzFqWAQG4Qq7XN0vumsMxwSjS) | [Baidu pan](https://pan.baidu.com/s/108SjV-uIJpxnqDMT19v-Aw)). Place them into `./models/sppe` and `./models/yolo` respectively.


## Quick Start
- **Input dir**:  Run AlphaPose for all images in a folder with:
```
python3 demo.py --indir ${img_directory} --outdir examples/res 
```
- **Video**:  Run AlphaPose for a video and save the rendered video with:
```
python3 video_demo.py --video ${path to video} --outdir examples/res --save_video
```
- **Webcam**:  Run AlphaPose using webcam and visualize the results with:
```
python3 webcam_demo.py --webcam 0 --outdir examples/res --vis
```
- **Input list**:  Run AlphaPose for images in a list and save the rendered images with:
```
python3 demo.py --list examples/list-coco-demo.txt --indir ${img_directory} --outdir examples/res --save_img
```
- **Note**:  If you meet OOM(out of memory) problem, decreasing the pose estimation batch until the program can run on your computer:
```
python3 demo.py --indir ${img_directory} --outdir examples/res --posebatch 30
```
- **Speeding up**:  Checkout the [speed_up.md](doc/speed_up.md) for more details.
- **For more**:  Checkout the [run.md](doc/run.md) for more options

## FAQ
Check out [faq.md](doc/faq.md) for faq.

## Contributors
Pytorch version of AlphaPose is developed and maintained by [Jiefeng Li](http://jeff-leaf.site/), [Hao-Shu Fang](https://fang-haoshu.github.io/) and [Cewu Lu](http://www.mvig.org/). 

## Citation
Please cite these papers in your publications if it helps your research:

    @inproceedings{fang2017rmpe,
      title={{RMPE}: Regional Multi-person Pose Estimation},
      author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
      booktitle={ICCV},
      year={2017}
    }

    @ARTICLE{2018arXiv180200977X,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      journal = {ArXiv e-prints},
      eprint = {1802.00977},
      year = {2018}
    }



## License
AlphaPose is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, contact [Cewu Lu](http://www.mvig.org/)
