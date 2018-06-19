import torch

from PIL import Image

import torchvision.transforms as transforms

from torchcv.transforms import resize
from torchcv.transforms import random_flip
from torchcv.transforms import random_paste

from torchcv.datasets import ListDataset
from torchcv.visualizations import vis_image
from torchcv.models.retinanet import BoxCoder

'''
ListDataset.

Put all image/box transformation in a function.
'''
def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=600)
    img, boxes = random_flip(img, boxes)
    img = transforms.ToTensor()(img)
    return img, boxes, labels

dataset = ListDataset(root='/mnt/hgfs/D/mscoco/2017/val2017',
                      list_file='torchcv/datasets/mscoco/coco17_val.txt',
                      transform=transform)

imgs, boxes, labels = dataset[0]
print(imgs.size())
print(boxes.size())
print(labels.size())

'''
BoxCoder in ListDataset transform.

If you resize/pad all images to the same size,
you can put BoxCoder directly into transform.
'''
box_coder = BoxCoder()
def transform_with_box_coder(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(600,600))  # resize all images to (600,600)
    img, boxes = random_flip(img, boxes)
    img = transforms.ToTensor()(img)

    boxes, labels = box_coder.encode(boxes, labels, (600,600))  # encode boxes here
    return img, boxes, labels

dataset = ListDataset(root='/mnt/hgfs/D/mscoco/2017/val2017',
                      list_file='torchcv/datasets/mscoco/coco17_val.txt',
                      transform=transform_with_box_coder)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
for imgs, boxes, labels in dataloader:
    print(imgs.size())
    print(boxes.size())
    print(labels.size())
    break

'''
BoxCoder in DataLoader collate_fn.

If you want to pad images to the max image size of the batch,
you can put BoxCoder in DataLoader collate_fn.
'''
def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=600)
    img, boxes = random_flip(img, boxes)
    return img, boxes, labels  # img is still PIL.Image

dataset = ListDataset(root='/mnt/hgfs/D/mscoco/2017/val2017',
                      list_file='torchcv/datasets/mscoco/coco17_val.txt',
                      transform=transform)

box_coder = BoxCoder()
def collate_fn(batch):
    imgs = [x[0] for x in batch]  # [PIL.Image]
    boxes = [x[1] for x in batch]
    labels = [x[2] for x in batch]

    num_imgs = len(imgs)
    max_w = max([im.size[0] for im in imgs])
    max_h = max([im.size[1] for im in imgs])
    inputs = torch.zeros(num_imgs, 3, max_h, max_w)

    loc_targets = []
    cls_targets = []
    for i in range(num_imgs):
        img = imgs[i]
        boxes_i = boxes[i].clone()
        labels_i = labels[i].clone()

        img, boxes_i = random_paste(img, boxes_i, (max_w,max_h))
        vis_image(img, boxes_i)
        inputs[i] = transforms.ToTensor()(img)

        loc_target, cls_target = box_coder.encode(boxes_i, labels_i, (max_w,max_h))
        loc_targets.append(loc_target)
        cls_targets.append(cls_target)
    return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate_fn)
for imgs, boxes, labels in dataloader:
    print(imgs.size())
    print(boxes.size())
    print(labels.size())
