import torch
import torchvision
import matplotlib.pyplot as plt


def vis_image(img, boxes=None, label_names=None, scores=None):
    '''Visualize a color image.

    Args:
      img: (PIL.Image/tensor) image to visualize.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      label_names: (list) label names.
      scores: (list) confidence scores.

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_bbox.py
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_image.py
    '''
    # Plot image
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if isinstance(img, torch.Tensor):
        img = torchvision.transforms.ToPILImage()(img)
    ax.imshow(img)

    # Plot boxes
    if boxes is not None:
        for i, bb in enumerate(boxes):
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0] + 1
            height = bb[3] - bb[1] + 1

            ax.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor='red', linewidth=2))

            caption = []
            if label_names is not None:
                caption.append(label_names[i])

            if scores is not None:
                caption.append('{:.2f}'.format(scores[i]))

            if len(caption) > 0:
                ax.text(bb[0], bb[1],
                        ': '.join(caption),
                        style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    # Show
    plt.show()
