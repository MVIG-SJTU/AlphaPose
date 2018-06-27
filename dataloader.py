import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from SPPE.src.utils.img import load_image, cropBox
from opt import opt


class Image_loader(data.Dataset):
    def __init__(self, img_list):
        if opt.imgpath:
            self.img_dir = opt.imgpath
        else:
            self.img_dir = './data/coco'
        self.imglist = open(img_list, 'r').readlines()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    '''# For faster rcnn
    def __getitem__(self, index):
        im_name = self.imglist[index].rstrip('\n').rstrip('\r')
        im_name = os.path.join(self.img_dir, im_name)
        im = cv2.imread(im_name)
        inp = load_image(im_name)
        return im, inp, im_name
    '''

    def __getitem__(self, index):
        im_name = self.imglist[index].rstrip('\n').rstrip('\r')
        im_name = os.path.join(self.img_dir, im_name)
        im = Image.open(im_name)
        inp = load_image(im_name)
        if im.mode == 'L':
            im = im.convert('RGB')

        ow = oh = 512
        im = im.resize((ow, oh))
        im = self.transform(im)
        return im, inp, im_name

    def __len__(self):
        # return len(self.imglist[0].rstrip('\n').rstrip('\r').split(','))
        return len(self.imglist)


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def crop_from_dets(img, boxes):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    inp = []
    pt1 = []
    pt2 = []
    for box in boxes:
        tmp_img = img.clone()
        tmp_img[0].add_(-0.406)
        tmp_img[1].add_(-0.457)
        tmp_img[2].add_(-0.480)

        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        tmp_inp = cropBox(tmp_img, upLeft, bottomRight, opt.inputResH, opt.inputResW)
        inp.append(tmp_inp.unsqueeze(0))
        pt1.append(upLeft.unsqueeze(0))
        pt2.append(bottomRight.unsqueeze(0))

    inp = torch.cat(inp, 0)
    pt1 = torch.cat(pt1, 0)
    pt2 = torch.cat(pt2, 0)

    return inp, pt1, pt2
