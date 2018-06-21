
<div align="center">
    <img src="doc/logo.jpg", width="400">
</div>


## News!

This is the **beta pytorch** version of AlphaPose. Stable version will be ready in two days. Currently AlphaPose runs at about **5 fps**. Realtime version is coming very soon. Stay tuned!

## AlphaPose
[Alpha Pose](http://www.mvig.org/research/alphapose.html) is an accurate multi-person pose estimator, which is the **first open-source system that achieves 70+ mAP (72.3 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset.** 
To match poses that correspond to the same person across frames, we also provide an efficient online pose tracker called Pose Flow. It is the **first open-source online pose tracker that achieves both 60+ mAP (66.5 mAP) and 50+ MOTA (58.3 MOTA) on PoseTrack Challenge dataset.**




## Installation
1. Get the code.
  ```Shell
  git clone -b pytorch https://github.com/MVIG-SJTU/AlphaPose.git

  ```

2. Install [pytorch](https://github.com/pytorch/pytorch)
  ```Shell
  chmod +x install.sh
  ./install.sh
  ```

1. Download the models manually: **fpnssd512_20_trained.pth**([Google Drive](https://drive.google.com/open?id=1KUk3WIFjXBDmNJYXZcdAb9S1k0fWmOMB) | [Baidu pan](https://pan.baidu.com/s/10ZQfHAqvn8SdFnPnuEg0fg)), **pyra_4.pth** ([Google Drive](https://drive.google.com/open?id=1oG1Fxj4oBfKwD1W_2QObxltWybuIk7Y6) | [Baidu pan](https://pan.baidu.com/s/14ONL_T_d1twm9Lxac5x-Ew)). Place them into `./models/ssd` and `./models/sppe` respectively.


## Quick Start
- **Demo**:  Run AlphaPose for all images in a folder and visualize the results with:
```
python demo_fast.py \
--inputlist ./list-coco-minival500.txt \
--imgpath ${img_directory} \
--outputpath ./coco-minival
```



## Contributors
AlphaPose is based on RMPE(ICCV'17), authored by [Hao-shu Fang](https://fang-haoshu.github.io/), Shuqin Xie, [Yu-Wing Tai](https://scholar.google.com/citations?user=nFhLmFkAAAAJ&hl=en) and [Cewu Lu](http://www.mvig.org/), [Cewu Lu](http://mvig.sjtu.edu.cn/) is the corresponding author. Currently, it is developed and maintained by [Hao-shu Fang](https://fang-haoshu.github.io/), [Jiefeng Li](http://jeff-leaf.site/), [Yuliang Xiu](http://xiuyuliang.cn/about/) and [Ruiheng Chang](https://crh19970307.github.io/). 

The main contributors are listed in [doc/contributors.md](doc/contributors.md).

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
