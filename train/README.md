### Train
This folder includes Torch code for training of the single person pose estimation network.

To run this code, the following must be installed:
* Torch7
* hdf5 (and the torch-hdf5 package)
* cudnn

We finetune our model based on the pre-trained pyraNet model.
``` bash
# First link the images dir
cd $SPPE_ROOT/train/src
ln -s /data/MPII/images ../data/mpii/images
ln -s /data/COCO/images ../data/coco/images
# Then finetune the model using PGPG
th main.lua -expID finetune -usePGPG -loadModel $MODEL_DIR -LR 0.5e-4 -nEpochs 50
# Or start a new training with multi-GPU
th main.lua -expID coco -nGPU 8 -trainBatch 16 -validBatch 16
```

For more options, you can read the `src/opt.lua` file.

