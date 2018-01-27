# faster-rcnn
A Tensorflow implementation of faster RCNN detection framework with soft-nms. 

### Detection Performance
We use ResNet152 as our backbone. Combined with multi-scale testing and soft-nms, the detection accuracy on COCO test-dev set is around **39.7**. We do not finetune the detector on person category since we found that it made no differences on the final pose estimation performance.

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/endernewton/tf-faster-rcnn.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd tf-faster-rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs.


3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

4. Install the [Python COCO API](https://github.com/pdollar/coco). The code requires the API to access COCO dataset.
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..
  ```
### Demo and Test with pre-trained models
1. Download pre-trained model
  ```Shell
  # Resnet101 for voc pre-trained on 07+12 set
  ./data/scripts/fetch_faster_rcnn_models.sh
  ```
  **Note**: if you cannot download the models through the link, or you want to try more models, you can check out the following solutions and optionally update the downloading script:
  - Another server [here](http://gs11655.sp.cs.cmu.edu/xinleic/tf-faster-rcnn/).
  - Google drive [here](https://drive.google.com/open?id=0B1_fAEgxdnvJSmF3YUlZcHFqWTQ).

2. Create a folder and a softlink to use the pre-trained model
  ```Shell
  NET=res101
  TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
  mkdir -p output/${NET}/${TRAIN_IMDB}
  cd output/${NET}/${TRAIN_IMDB}
  ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
  cd ../../..
  ```

3. Demo for testing on custom images
  ```Shell
  # at reposistory root
  GPU_ID=0
  CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
  ```
  **Note**: Resnet101 testing probably requires several gigabytes of memory, so if you encounter memory capacity issues, please install it with CPU support only. Refer to [Issue 25](https://github.com/endernewton/tf-faster-rcnn/issues/25).

4. Test with pre-trained Resnet101 models
  ```Shell
  GPU_ID=0
  ./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101
  ```
  **Note**: If you cannot get the reported numbers (78.7 on my side), then probabaly the NMS function is compiled improperly, refer to [Issue 5](https://github.com/endernewton/tf-faster-rcnn/issues/5).

