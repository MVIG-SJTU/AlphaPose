## Installation

### Requirements
* Nvidia device with CUDA, [example for Ubuntu 20.04](https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux)
(if you have no nvidia device, delete [this line](https://github.com/MVIG-SJTU/AlphaPose/blob/master/setup.py#L211) from setup.py
* Python 3.5+
* Cython
* PyTorch 1.1+, for users with PyTorch 1.5 and 1.5+, please merge the pull request #592 by:
  `git pull origin pull/592/head`
* torchvision 0.3.0+
* numpy 
* python-package setuptools >= 40.0, reported by [this issue](https://github.com/MVIG-SJTU/AlphaPose/issues/838)
* Linux, [Windows user check here](#Windows)

### Code installation

#### (Recommended) Install with conda

Install conda from [here](https://repo.anaconda.com/miniconda/), Miniconda3-latest-(OS)-(platform).
```shell
# 1. Create a conda virtual environment.
conda create -n alphapose python=3.6 -y
conda activate alphapose

# 2. Install PyTorch
conda install pytorch==1.1.0 torchvision==0.3.0

# 3. Get AlphaPose
git clone https://github.com/MVIG-SJTU/AlphaPose.git
# git pull origin pull/592/head if you use PyTorch>=1.5
cd AlphaPose


# 4. install
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
python -m pip install cython
sudo apt-get install libyaml-dev
################Only For Ubuntu 18.04#################
locale-gen C.UTF-8
# if locale-gen not found
sudo apt-get install locales
export LANG=C.UTF-8
######################################################
python setup.py build develop
```

#### Install with pip
```shell
# 1. Install PyTorch
pip3 install torch==1.1.0 torchvision==0.3.0

# Check torch environment by:  python3 -m torch.utils.collect_env

# 2. Get AlphaPose
git clone https://github.com/MVIG-SJTU/AlphaPose.git
# git pull origin pull/592/head if you use PyTorch>=1.5
cd AlphaPose

# 3. install
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
pip install cython
sudo apt-get install libyaml-dev
python3 setup.py build develop --user
```

#### Windows
The installation process is same as above. But note that Windows users may face problem when installing cuda extension. Thus we disable the cuda extension in the setup.py by default. The affect is that models ended with "-dcn" is not supported. If you force to make cuda extension by modify [this line](https://github.com/MVIG-SJTU/AlphaPose/blob/master/setup.py#L124) to True, you should install Visual Studio due to the problem mentioned [here](https://github.com/MVIG-SJTU/AlphaPose/blob/master/setup.py#L121).
We recommend Windows users to run models like FastPose, FastPose-duc, etc., as they also provide good accuracy and speed.

For Windows user, if you meet error with PyYaml, you can download and install it manually from here: https://pyyaml.org/wiki/PyYAML.
If your OS platform is `Windows`, make sure that Windows C++ build tool like visual studio 15+ or visual c++ 2015+ is installed for training.

### Models
1. Download the object detection model manually: **yolov3-spp.weights**([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)). Place it into `detector/yolo/data`.

2. Download our pose models. Place them into `pretrained_models`. All models and details are available in our [Model Zoo](./MODEL_ZOO.md).

2. For pose tracking, please refer to our [tracking docments](../trackers) for model download



### Prepare dataset (optional)

#### MSCOCO
If you want to train the model by yourself, please download data from [MSCOCO](http://cocodataset.org/#download) (train2017 and val2017). Download and extract them under `./data`, and make them look like this:
```
|-- json
|-- exp
|-- alphapose
|-- configs
|-- test
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- train2017
        |   |-- 000000000009.jpg
        |   |-- 000000000025.jpg
        |   |-- 000000000030.jpg
        |   |-- ... 
        `-- val2017
            |-- 000000000139.jpg
            |-- 000000000285.jpg
            |-- 000000000632.jpg
            |-- ... 
```

#### MPII
Please download images from [MPII](http://human-pose.mpi-inf.mpg.de/#download). We also provide the annotations in json format [[annot_mpii.zip](https://drive.google.com/open?id=1HC6znReBeg-TMPZbmoldtYrMGlrEFamh)]. 
Download and extract them under `./data`, and make them look like this:
```
|-- data
`-- |-- mpii
    `-- |-- annot_mpii.json
        `-- images
            |-- 027457270.jpg
            |-- 036645665.jpg
            |-- 045572740.jpg
            |-- ... 
```

#### Halpe-FullBody
If you want to train the model by yourself, please download data from [Halpe-FullBody](https://github.com/Fang-Haoshu/Halpe-FullBody). Download and extract them under `./data`, and make them look like this:
```
|-- json
|-- exp
|-- alphapose
|-- configs
|-- test
|-- data
`-- |-- halpe
    `-- |-- annotations
        |   |-- halpe_train_v1.json
        |   `-- halpe_val_v1.json
        |-- images
        `-- |-- train2015
             |   |-- HICO_train2015_00000001.jpg
             |   |-- HICO_train2015_00000002.jpg
             |   |-- HICO_train2015_00000003.jpg
             |   |-- ... 
             `-- val2017
                 |-- 000000000139.jpg
                 |-- 000000000285.jpg
                 |-- 000000000632.jpg
                 |-- ... 
```
