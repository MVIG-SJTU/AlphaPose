from PIL import Image

from torchcv.datasets import ListDataset
from torchcv.visualizations import vis_image
from torchcv.transforms import resize, random_flip, random_crop, random_paste

import torchvision.transforms as transforms


def transform(img, boxes, labels):
    img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=600, random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.ToTensor()(img)
    return img, boxes, labels

dataset = ListDataset(root='/mnt/hgfs/D/mscoco/2017/val2017',
                      list_file='torchcv/datasets/mscoco/coco17_val.txt',
                      transform=transform)

img, boxes, labels = dataset[0]
vis_image(img, boxes)
