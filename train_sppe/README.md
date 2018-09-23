This folder includes PyTorch code for training the Single Person Pose Estimation network in AlphaPose.

## Installation
1. Install PyTorch >= 0.4.0 following [official instruction](https://pytorch.org/get-started/locally/).
2. Install other dependencies.
``` bash
cd ${TRAIN_ROOT}$
pip install -r requirements.txt
```
3. Disable CUDNN for batchnormalization
```
# PYTORCH=/path/to/pytorch
sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
```

## Data preparation

### COCO Data
Please download [annot_coco.h5](https://drive.google.com/open?id=1OviCQgzKO2t0gh4Me0MXfi6xgXyTWC5T) and `person_keypoints_val2014.json` from `cocodataset`(http://cocodataset.org/#download).
```
${TRAIN_ROOT}
|-- src
|-- exp
|-- data
`-- |-- coco
    `-- |-- annot_coco.h5
        |-- person_keypoints_val2014.json
        `-- images
            |-- trainval2017
            |   |-- 000000000001.jpg
            |   |-- 000000000002.jpg
            |   |-- 000000000003.jpg
            |   |-- ... 
```

## Train on COCO
``` bash
# Train without DPG first
python train.py --dataset coco --expID exp1 --nClasses 17 --LR 1e-4
# Then, train with DPG
python train.py --dataset coco --expID exp1 --nClasses 17 --LR 1e-4 --addDPG

# Or, train with pretrian model
python train.py --dataset coco --expID exp1 --nClasses 17 --LR 1e-5 --addDPG --loadModel #{MODEL_DIR}
```

