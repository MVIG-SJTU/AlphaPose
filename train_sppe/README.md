This folder includes PyTorch code for training the Single Person Pose Estimation network in AlphaPose.

## Installation
1. Install PyTorch >= 0.4.0 following [official instruction](https://pytorch.org/get-started/locally/).
2. Install other dependencies.
``` bash
cd ${TRAIN_ROOT}
pip install -r requirements.txt
```
3.Disable cudnn for batch_norm: (See: [@Microsoft / human-pose-estimation.pytorch#installation](https://github.com/Microsoft/human-pose-estimation.pytorch#installation))
```
# PYTORCH=/path/to/pytorch
# for pytorch v0.4.0
sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
# for pytorch v0.4.1
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py

# Note that instructions like # PYTORCH=/path/to/pytorch indicate that you should pick 
# a path where you'd like to have pytorch installed and then set an environment
# variable (PYTORCH in this case) accordingly.
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
cd src
# Train without DPG first
python train.py --dataset coco --expID exp1 --nClasses 17 --LR 1e-4
# Then, train with DPG
python train.py --dataset coco --expID exp1 --nClasses 17 --LR 1e-4 --addDPG

# Or, train with pretrian model
python train.py --dataset coco --expID exp1 --nClasses 17 --LR 1e-5 --addDPG --loadModel #{MODEL_DIR}
```

## Train on new dataset
Please create the `h5` file from your own datset. Here is the python demo to read the `h5` file.
``` python
>>> import h5py
>>> annot = h5py.File('annot_coco.h5')
>>> for k in annot.keys():
...     print(k)
bndbox
imgname
part

>>> bndboxes = annot['bndbox'][:]
>>> bndbox.shape
(144213, 1, 4)
>>> imgnames = annot['imgname'][:]
>>> imgname.shape
(144213, 16)
>>> parts = annot['part'][:]
>>> part.shape
(144213, 17, 2)
```
```
bndbox:     [1 x 4]     (upleft_x, upleft_y, bottomright_x, bottomright_y)
imgname:    [16]        #ascii number of imagename
part:       [17 x 2]    (kp1_x, kp1_y, kp2_x, kp2_y, ..., kp17_x, kp17_y)
```

Please refer to this python demo to create your own `h5` files. How to save data in `h5` files, please refer to [h5py quick start](http://docs.h5py.org/en/stable/quick.html#quick).
