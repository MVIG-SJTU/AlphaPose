
<div align="center">
    <img src="doc/logo.jpg", width="400">
</div>

## News!
- Apr 2019: [**MXNet** version](https://github.com/MVIG-SJTU/AlphaPose/tree/mxnet) of AlphaPose is released! It runs at **23 fps** on COCO validation set.
- Feb 2019: [CrowdPose](https://github.com/MVIG-SJTU/AlphaPose/blob/pytorch/doc/CrowdPose.md) is integrated into AlphaPose Now!
- Dec 2018: [General version](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch/PoseFlow) of PoseFlow is released! 3X Faster and support pose tracking results visualization!
- Sep 2018: [**PyTorch** version](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch) of AlphaPose is released! It runs at **20 fps** on COCO validation set (4.6 people per image on average) and achieves 71 mAP!

# AlphaPose
Gluon implementation of AlphaPose, faster and more accurate.

## Prerequisite
- GluonCV
- MXNet==1.3.0
- tqdm==4.19.1
- Matplotlib
- NumPy

## Usage
Download model weights from [Google Drive](https://drive.google.com/open?id=1TTf8Ox-ECGXRAeX4cHYkEMBDVJEZgBL6) and put it into [sppe/params](sppe/params).

### Image Demo
```bash
MXNET_CPU_WORKER_NTHREADS=2 python demo.py
```

### Video Demo
```bash
MXNET_CPU_WORKER_NTHREADS=2 python video_demo.py
```

## Contributors
MXNet version of AlphaPose is developed and maintained by [Jiefeng Li](http://jeff-leaf.site/), [Chenxi Wang](https://github.com/chenxi-wang), [Hao-Shu Fang](https://fang-haoshu.github.io/) and [Cewu Lu](http://www.mvig.org/). 


## Citation
Please cite these papers in your publications if it helps your research:

    @inproceedings{fang2017rmpe,
      title={{RMPE}: Regional Multi-person Pose Estimation},
      author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
      booktitle={ICCV},
      year={2017}
    }

    @inproceedings{xiu2018poseflow,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      booktitle={BMVC},
      year = {2018}
    }



## License
AlphaPose is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at mvig.alphapose[at]gmail[dot]com and cc lucewu[[at]sjtu[dot]edu[dot]cn. We will send the detail agreement to you.

