import numpy as np
import cv2
import torch
import scipy.misc
from torchvision import transforms
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

from PIL import Image
from copy import deepcopy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def im_to_torch(img):
    img=np.array(img)
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_im(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode='RGB'))


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def drawCircle(img, pt, sigma):
    img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g[g > 0] = 1
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)


def drawGaussian(img, pt, sigma):
    img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0    
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)


def drawBigCircle(img, pt, sigma):
    img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g[g > 0.4] = 1
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)


def drawSmallCircle(img, pt, sigma):
    img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g[g > 0.5] = 1
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)


def transformBox(pt, ul, br, inpH, inpW, resH, resW):
    center = torch.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = torch.zeros(2)
    _pt[0] = pt[0] - ul[0]
    _pt[1] = pt[1] - ul[1]
    # Move to center
    _pt[0] = _pt[0] + max(0, (lenW - 1) / 2 - center[0])
    _pt[1] = _pt[1] + max(0, (lenH - 1) / 2 - center[1])
    pt = (_pt * resH) / lenH
    pt[0] = round(float(pt[0]))
    pt[1] = round(float(pt[1]))
    return pt.int()


def transformBoxInvert(pt, ul, br, inpH, inpW, resH, resW):
    center = np.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = (pt * lenH) / resH
    _pt[0] = _pt[0] - max(0, (lenW - 1) / 2 - center[0])
    _pt[1] = _pt[1] - max(0, (lenH - 1) / 2 - center[1])

    new_point = np.zeros(2)
    new_point[0] = _pt[0] + ul[0]
    new_point[1] = _pt[1] + ul[1]
    return new_point


def transformBoxInvert_batch(pt, ul, br, inpH, inpW, resH, resW):
    '''
    pt:     [n, 17, 2]
    ul:     [n, 2]
    br:     [n, 2]
    '''
    center = (br - 1 - ul) / 2

    size = br - ul
    size[:, 0] *= (inpH / inpW)

    lenH, _ = torch.max(size, dim=1)   # [n,]
    lenW = lenH * (inpW / inpH)

    _pt = (pt * lenH[:, np.newaxis, np.newaxis]) / resH
    _pt[:, :, 0] = _pt[:, :, 0] - ((lenW[:, np.newaxis].repeat(1, 17) - 1) /
                                   2 - center[:, 0].unsqueeze(-1).repeat(1, 17)).clamp(min=0)
    _pt[:, :, 1] = _pt[:, :, 1] - ((lenH[:, np.newaxis].repeat(1, 17) - 1) /
                                   2 - center[:, 1].unsqueeze(-1).repeat(1, 17)).clamp(min=0)

    new_point = torch.zeros(pt.size())
    new_point[:, :, 0] = _pt[:, :, 0] + ul[:, 0].unsqueeze(-1).repeat(1, 17)
    new_point[:, :, 1] = _pt[:, :, 1] + ul[:, 1].unsqueeze(-1).repeat(1, 17)
    return new_point


def cropBox(img, ul, br, resH, resW):
    ul = ul.int()
    br = (br - 1).int()
    # br = br.int()
    lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    box_shape = [(br[1] - ul[1]).item(), (br[0] - ul[0]).item()]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    if ul[1] > 0:
        img[:, :ul[1], :] = 0
    if ul[0] > 0:
        img[:, :, :ul[0]] = 0
    if br[1] < img.shape[1] - 1:
        img[:, br[1] + 1:, :] = 0
    if br[0] < img.shape[2] - 1:
        img[:, :, br[0] + 1:] = 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array(
        [ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :] = np.array(
        [br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

    return im_to_torch(torch.Tensor(dst_img))


def cv_rotate(img, rot, resW, resH):
    center = np.array((resW - 1, resH - 1)) / 2
    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, (resH - 1) * -0.5], rot_rad)
    dst_dir = np.array([0, (resH - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [(resW - 1) * 0.5, (resH - 1) * 0.5]
    dst[1, :] = np.array([(resW - 1) * 0.5, (resH - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

    return im_to_torch(torch.Tensor(dst_img))


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1
    if '0.4.1' in torch.__version__ or '1.0' in torch.__version__:
        return x.flip(dims=(dim,))
    else:
        is_cuda = False
        if x.is_cuda:
            is_cuda = True
            x = x.cpu()
        x = x.numpy().copy()
        if x.ndim == 3:
            x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
        elif x.ndim == 4:
            for i in range(x.shape[0]):
                x[i] = np.transpose(
                    np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
        # x = x.swapaxes(dim, 0)
        # x = x[::-1, ...]
        # x = x.swapaxes(0, dim)

        x = torch.from_numpy(x.copy())
        if is_cuda:
            x = x.cuda()
        return x


def shuffleLR(x, dataset):
    flipRef = dataset.flipRef
    assert (x.dim() == 3 or x.dim() == 4)
    for pair in flipRef:
        dim0, dim1 = pair
        dim0 -= 1
        dim1 -= 1
        if x.dim() == 4:
            tmp = x[:, dim1].clone()
            x[:, dim1] = x[:, dim0].clone()
            x[:, dim0] = tmp.clone()
            #x[:, dim0], x[:, dim1] = deepcopy((x[:, dim1], x[:, dim0]))
        else:
            tmp = x[dim1].clone()
            x[dim1] = x[dim0].clone()
            x[dim0] = tmp.clone()
            #x[dim0], x[dim1] = deepcopy((x[dim1], x[dim0]))
    return x


def drawMPII(inps, preds):
    assert inps.dim() == 4
    p_color = ['g', 'b', 'purple', 'b', 'purple',
               'y', 'o', 'y', 'o', 'y', 'o',
               'pink', 'r', 'pink', 'r', 'pink', 'r']
    p_color = ['r', 'r', 'r', 'b', 'b', 'b',
               'black', 'black', 'black', 'black',
               'y', 'y', 'white', 'white', 'g', 'g']

    nImg = inps.size(0)
    imgs = []
    for n in range(nImg):
        img = to_numpy(inps[n])
        img = np.transpose(img, (1, 2, 0))
        imgs.append(img)

    fig = plt.figure()
    plt.imshow(imgs[0])
    ax = fig.add_subplot(1, 1, 1)
    #print(preds.shape)
    for p in range(16):
        x, y = preds[0][p]
        cor = (round(x), round(y)), 10
        ax.add_patch(plt.Circle(*cor, color=p_color[p]))
    plt.axis('off')

    plt.show()

    return imgs


def drawCOCO(inps, preds, scores):
    assert inps.dim() == 4
    p_color = ['g', 'b', 'purple', 'b', 'purple',
               'y', 'orange', 'y', 'orange', 'y', 'orange',
               'pink', 'r', 'pink', 'r', 'pink', 'r']

    nImg = inps.size(0)
    imgs = []
    for n in range(nImg):
        img = to_numpy(inps[n])
        img = np.transpose(img, (1, 2, 0))
        imgs.append(img)

    fig = plt.figure()
    plt.imshow(imgs[0])
    ax = fig.add_subplot(1, 1, 1)
    #print(preds.shape)
    for p in range(17):
        if scores[0][p][0] < 0.2:
            continue
        x, y = preds[0][p]
        cor = (round(x), round(y)), 3
        ax.add_patch(plt.Circle(*cor, color=p_color[p]))
    plt.axis('off')

    plt.show()

    return imgs


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def findPeak(hm):
    mx = maximum_filter(hm, size=5)
    idx = zip(*np.where((mx == hm) * (hm > 0.1)))
    candidate_points = []
    for (y, x) in idx:
        candidate_points.append([x, y, hm[y][x]])
    if len(candidate_points) == 0:
        return torch.zeros(0)
    candidate_points = np.array(candidate_points)
    candidate_points = candidate_points[np.lexsort(-candidate_points.T)]
    return torch.Tensor(candidate_points)


def processPeaks(candidate_points, hm, pt1, pt2, inpH, inpW, resH, resW):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> List[Tensor]

    if candidate_points.shape[0] == 0:  # Low Response
        maxval = np.max(hm.reshape(1, -1), 1)
        idx = np.argmax(hm.reshape(1, -1), 1)

        x = idx % resW
        y = int(idx / resW)

        candidate_points = np.zeros((1, 3))
        candidate_points[0, 0:1] = x
        candidate_points[0, 1:2] = y
        candidate_points[0, 2:3] = maxval

    res_pts = []
    for i in range(candidate_points.shape[0]):
        x, y, maxval = candidate_points[i][0], candidate_points[i][1], candidate_points[i][2]

        if bool(maxval < 0.05) and len(res_pts) > 0:
            pass
        else:
            if bool(x > 0) and bool(x < resW - 2):
                if bool(hm[int(y)][int(x) + 1] - hm[int(y)][int(x) - 1] > 0):
                    x += 0.25
                elif bool(hm[int(y)][int(x) + 1] - hm[int(y)][int(x) - 1] < 0):
                    x -= 0.25
            if bool(y > 0) and bool(y < resH - 2):
                if bool(hm[int(y) + 1][int(x)] - hm[int(y) - 1][int(x)] > 0):
                    y += (0.25 * inpH / inpW)
                elif bool(hm[int(y) + 1][int(x)] - hm[int(y) - 1][int(x)] < 0):
                    y -= (0.25 * inpH / inpW)

            #pt = torch.zeros(2)
            pt = np.zeros(2)
            pt[0] = x + 0.2
            pt[1] = y + 0.2

            pt = transformBoxInvert(pt, pt1, pt2, inpH, inpW, resH, resW)

            res_pt = np.zeros(3)
            res_pt[:2] = pt
            res_pt[2] = maxval

            res_pts.append(res_pt)

            if maxval < 0.05:
                break
    return res_pts
