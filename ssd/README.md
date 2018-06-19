# TorchCV: a PyTorch vision library mimics ChainerCV


## Detection
| Model                | Original Paper | ChainerCV  | TorchCV*  |
| -------------------- | -------------- | ---------- | -------   |
| SSD300@voc07_test    | 74.3%          | 77.8%      |  76.68%   |
| SSD512@voc07_test    | 76.8%          | 79.2%      |  78.89%   |
| FPNSSD512@voc07_test | -              | -          |  81.46%   |

The accuracy of TorchCV SSD is ~1% lower than ChainerCV. This is because the VGG base model I use performs slightly worse.  
I did the experiment by replacing [pytorch/vision](https://github.com/pytorch/vision) VGG16 model with the [model](https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/ssd_vgg16.py#L298) used in ChainerCV, the SSD512 model got __79.85%__ accuracy.

FPNSSD512 is created by replacing SSD VGG16 network with FPN50, the rest is the same. It beats all SSD models.  
You can download the trained params [here](https://drive.google.com/open?id=1yy_kUnm_hZR3uk9yLcaQSMwxVn7wApTU).

## Update
[2018-2-6] Our FPNSSD512 model achieved the 1st place on the PASCAL VOC 2012 dataset.

![image](https://user-images.githubusercontent.com/10502826/35849236-d8f0ec26-0b5b-11e8-8e13-15b08ac40c8d.png)
Check the [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=3#KEY_FPNSSD).

[2018-2-26] As [issue(#11)](https://github.com/kuangliu/torchcv/issues/11) mentioned I shouldn't use VOC07 data for training. I submit another result that is only trained on VOC12 data. The older submission is already marked to private.

![image](https://user-images.githubusercontent.com/10502826/36658797-1938173a-1b0d-11e8-810f-b1b85662ce4b.png)

[2018-3-29] As Alibaba Turing Lab submit a result of __74.8%__ MAP, which takes the first place on Comp3, I decided to train a deeper model (replace FPN50 with FPN152, trained only with VOC12 data).  
It got MAP of __77%__, which is far more higher than I expected.  
Check the new [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=3#KEY_FPNSSD). The older submission is marked to private.

## TODO
- [x] SSD300
- [x] SSD512
- [x] FPNSSD512
- [ ] RetinaNet
- [ ] Faster R-CNN
- [ ] Mask R-CNN
