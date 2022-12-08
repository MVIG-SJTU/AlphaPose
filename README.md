
<div align="center">
    <img src="docs/logo.jpg", width="400">
</div>


## News!
- Nov 2022: [**AlphaPose paper**](http://arxiv.org/abs/2211.03375) is released! Checkout the paper for more details about this project.
- Sep 2022: [**Jittor** version](https://github.com/tycoer/AlphaPose_jittor) of AlphaPose is released! It achieves 1.45x speed up with resnet50 backbone on the training stage.
- July 2022: [**v0.6.0** version](https://github.com/MVIG-SJTU/AlphaPose) of AlphaPose is released! [HybrIK](https://github.com/Jeff-sjtu/HybrIK) for 3D pose and shape estimation is supported!
- Jan 2022: [**v0.5.0** version](https://github.com/MVIG-SJTU/AlphaPose) of AlphaPose is released! Stronger whole body(face,hand,foot) keypoints! More models are availabel. Checkout [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md)
- Aug 2020: [**v0.4.0** version](https://github.com/MVIG-SJTU/AlphaPose) of AlphaPose is released! Stronger tracking! Include whole body(face,hand,foot) keypoints! [Colab](https://colab.research.google.com/drive/1c7xb_7U61HmeJp55xjXs24hf1GUtHmPs?usp=sharing) now available.
- Dec 2019: [**v0.3.0** version](https://github.com/MVIG-SJTU/AlphaPose) of AlphaPose is released! Smaller model, higher accuracy!
- Apr 2019: [**MXNet** version](https://github.com/MVIG-SJTU/AlphaPose/tree/mxnet) of AlphaPose is released! It runs at **23 fps** on COCO validation set.
- Feb 2019: [CrowdPose](https://github.com/MVIG-SJTU/AlphaPose/docs/CrowdPose.md) is integrated into AlphaPose Now!
- Dec 2018: [General version](https://github.com/MVIG-SJTU/AlphaPose/trackers/PoseFlow) of PoseFlow is released! 3X Faster and support pose tracking results visualization!
- Sep 2018: [**v0.2.0** version](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch) of AlphaPose is released! It runs at **20 fps** on COCO validation set (4.6 people per image on average) and achieves 71 mAP!

## AlphaPose
[AlphaPose](http://www.mvig.org/research/alphapose.html) is an accurate multi-person pose estimator, which is the **first open-source system that achieves 70+ mAP (75 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset.** 
To match poses that correspond to the same person across frames, we also provide an efficient online pose tracker called Pose Flow. It is the **first open-source online pose tracker that achieves both 60+ mAP (66.5 mAP) and 50+ MOTA (58.3 MOTA) on PoseTrack Challenge dataset.**

AlphaPose supports both Linux and **Windows!**

<div align="center">
    <img src="docs/alphapose_17.gif", width="400" alt><br>
    COCO 17 keypoints
</div>
<div align="center">
    <img src="docs/alphapose_26.gif", width="400" alt><br>
    <b><a href="https://github.com/Fang-Haoshu/Halpe-FullBody">Halpe 26 keypoints</a></b> + tracking
</div>
<div align="center">
    <img src="docs/alphapose_136.gif", width="400"alt><br>
    <b><a href="https://github.com/Fang-Haoshu/Halpe-FullBody">Halpe 136 keypoints</a></b> + tracking
</div>
<div align="center">
    <img src="docs/alphapose_hybrik_smpl.gif", width="400"alt><br>
    <b><a href="https://github.com/Jeff-sjtu/HybrIK">SMPL</a></b> + tracking
</div>


## Results
### Pose Estimation
Results on COCO test-dev 2015:
<center>

| Method | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|
| OpenPose (CMU-Pose) | 61.8 | 84.9 | 67.5 | 57.1 | 68.2 |
| Detectron (Mask R-CNN) | 67.0 | 88.0 | 73.1 | 62.2 | 75.6 |
| **AlphaPose** | **73.3** | **89.2** | **79.1** | **69.0** | **78.6** |

</center>

Results on MPII full test set:
<center>

| Method | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Ave |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| OpenPose (CMU-Pose) | 91.2 | 87.6 | 77.7 | 66.8 | 75.4 | 68.9 | 61.7 | 75.6 |
| Newell & Deng | **92.1** | 89.3 | 78.9 | 69.8 | 76.2 | 71.6 | 64.7 | 77.5 |
| **AlphaPose** | 91.3 | **90.5** | **84.0** | **76.4** | **80.3** | **79.9** | **72.4** | **82.1** |

</center>

More results and models are available in the [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md).

### Pose Tracking

<p align='center'>
    <img src="docs/posetrack.gif", width="360">
    <img src="docs/posetrack2.gif", width="344">
</p>

Please read [trackers/README.md](trackers/) for details.

### CrowdPose
<p align='center'>
    <img src="docs/crowdpose.gif", width="360">
</p>

Please read [docs/CrowdPose.md](docs/CrowdPose.md) for details.


## Installation
Please check out [docs/INSTALL.md](docs/INSTALL.md)

## Model Zoo
Please check out [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md)

## Quick Start
- **Colab**: We provide a [colab example](https://colab.research.google.com/drive/1_3Wxi4H3QGVC28snL3rHIoeMAwI2otMR?usp=sharing) for your quick start.

- **Inference**: Inference demo
``` bash
./scripts/inference.sh ${CONFIG} ${CHECKPOINT} ${VIDEO_NAME} # ${OUTPUT_DIR}, optional
```

Inference SMPL (Download the SMPL model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) and put it in `model_files/`).
``` bash
./scripts/inference_3d.sh ./configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml ${CHECKPOINT} ${VIDEO_NAME} # ${OUTPUT_DIR}, optional
```
For high level API, please refer to `./scripts/demo_api.py`. To enable tracking, please refer to [this page](./trackers).

- **Training**: Train from scratch
``` bash
./scripts/train.sh ${CONFIG} ${EXP_ID}
```

- **Validation**: Validate your model on MSCOCO val2017
``` bash
./scripts/validate.sh ${CONFIG} ${CHECKPOINT}
```

Examples:

Demo using `FastPose` model.
``` bash
./scripts/inference.sh configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml pretrained_models/fast_res50_256x192.pth ${VIDEO_NAME}
#or
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
#or if you want to use yolox-x as the detector
python scripts/demo_inference.py --detector yolox-x --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
```

Train `FastPose` on mscoco dataset.
``` bash
./scripts/train.sh ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml exp_fastpose
```

More detailed inference options and examples, please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md)


## Common issue & FAQ
Check out [faq.md](docs/faq.md) for faq. If it can not solve your problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request!

## Contributors
AlphaPose is based on RMPE(ICCV'17), authored by [Hao-Shu Fang](https://fang-haoshu.github.io/), Shuqin Xie, [Yu-Wing Tai](https://scholar.google.com/citations?user=nFhLmFkAAAAJ&hl=en) and [Cewu Lu](http://www.mvig.org/), [Cewu Lu](http://mvig.sjtu.edu.cn/) is the corresponding author. Currently, it is maintained by [Jiefeng Li\*](http://jeff-leaf.site/), [Hao-shu Fang\*](https://fang-haoshu.github.io/),  [Haoyi Zhu](https://github.com/HaoyiZhu), [Yuliang Xiu](http://xiuyuliang.cn/about/) and [Chao Xu](http://www.isdas.cn/). 

The main contributors are listed in [doc/contributors.md](docs/contributors.md).

## TODO
- [x] Multi-GPU/CPU inference
- [x] 3D pose
- [x] add tracking flag
- [ ] PyTorch C++ version
- [x] Add model trained on mixture dataset (Check the model zoo)
- [ ] dense support
- [x] small box easy filter
- [x] Crowdpose support
- [ ] Speed up PoseFlow
- [x] Add stronger/light detectors (yolox is now supported)
- [x] High level API (check the scripts/demo_api.py)

We would really appreciate if you can offer any help and be the [contributor](docs/contributors.md) of AlphaPose.


## Citation
Please cite these papers in your publications if it helps your research:

    @article{alphapose,
      author = {Fang, Hao-Shu and Li, Jiefeng and Tang, Hongyang and Xu, Chao and Zhu, Haoyi and Xiu, Yuliang and Li, Yong-Lu and Lu, Cewu},
      journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
      title = {AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time},
      year = {2022}
    }
    
    @inproceedings{fang2017rmpe,
      title={{RMPE}: Regional Multi-person Pose Estimation},
      author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
      booktitle={ICCV},
      year={2017}
    }

    @inproceedings{li2019crowdpose,
        title={Crowdpose: Efficient crowded scenes pose estimation and a new benchmark},
        author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
        booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
        pages={10863--10872},
        year={2019}
    }

If you used the 3D mesh reconstruction module, please also cite:

    @inproceedings{li2021hybrik,
        title={Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation},
        author={Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={3383--3393},
        year={2021}
    }

If you used the PoseFlow tracking module, please also cite:

    @inproceedings{xiu2018poseflow,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      booktitle={BMVC},
      year = {2018}
    }





## License
AlphaPose is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at mvig.alphapose[at]gmail[dot]com and cc lucewu[[at]sjtu[dot]edu[dot]cn. We will send the detail agreement to you.
