### Train
We finetune our model based on the pre-trained pyraNet model.
``` bash
cd $SPPE_ROOT/train/src
# First finetune the model using DPG
th main.lua -expID finetune -addDPG
th main.lua -expID finetune -addDPG -continue -LR 0.5e-4 -nEpochs 10
th main.lua -expID final_model -loadModel '../exp/coco/finetune/final_model.t7' -LR 0.5e-4 -addDPG
th main.lua -expID final_model -continue -nEpochs 10 -LR 0.1e-4 -addDPG
# You can create a new model with different input size, heatmap size and other PRM options
th main.lua -expID new_model -nEpochs 40 -LR 2.5e-4 -inputResH 384 -inputResW 256 -outputResH 96 -outputResW 64 -baseWidth 6 -cardinality 5
```

### Evaluate
You will need to generate bounding box first. Here we have already generated the bounding boxes in `$SPPE_ROOT/predict/annot/coco-test/`. To generate it yourself, please follow the guidlines in the main repo.

``` bash
cd $SPPE_ROOT/predict
# make a soft link to the images
ln -s /data/COCO/images/ data/images
# get the predicted results
th main-alpha-pose.lua predict-test
# Or visulize the SPPE result with following command
qlua main-alpha-pose.lua demo
```