import os
import argparse
import logging
from types import MethodType


parser = argparse.ArgumentParser(description='Mxnet AlphaPose')

"----------------------------- Experiment options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--dataset', default='coco', type=str,
                    help='Dataset choice: mpii | coco')

"----------------------------- General options -----------------------------"
parser.add_argument('--nThreads', default=60, type=int,
                    help='Number of data loading threads')
parser.add_argument('--snapshot', default=1, type=int,
                    help='How often to take a snapshot of the model (0 = never)')
parser.add_argument('--load_from_pyt', default=False, dest='load_from_pyt',
                    help='Load pretrained model from PyTorch model', action='store_true')

"----------------------------- Training options -----------------------------"
parser.add_argument('--addDPG', default=False, dest='addDPG',
                    help='Train with data augmentation', action='store_true')
parser.add_argument('--hardMining', default=False, dest='hardMining',
                    help='Train with data augmentation', action='store_true')
parser.add_argument('--LR', default=1e-4, type=float,
                    help='Learning rate')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='epsilon')
parser.add_argument('--crit', default='MSE', type=str,
                    help='Criterion type (MSE, KLD, KLD2)')
parser.add_argument('--optMethod', default='rmsprop', type=str,
                    help='Optimization method: rmsprop | sgd | nag | adadelta')
parser.add_argument('--nEpochs', default=100, type=int,
                    help='Number of hourglasses to stack')
parser.add_argument('--trainBatch', default=28, type=int,
                    help='Train-batch size')
parser.add_argument('--validBatch', default=24, type=int,
                    help='Valid-batch size')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--gpu_id', default=[0], type=list,
                    help='GPU used')
parser.add_argument('--lr_decay_epoch', type=str, default='20,60',
                    help='epoches at which learning rate decays. default is 10,60.')

"----------------------------- Iteration options -----------------------------"
parser.add_argument('--epoch', default=0, type=int,
                    help='Current epoch')
parser.add_argument('--trainIters', default=0, type=int,
                    help='Total train iters')
parser.add_argument('--valIters', default=0, type=int,
                    help='Total valid iters')

"----------------------------- Model options -----------------------------"
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
parser.add_argument('--try_loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
parser.add_argument('--nClasses', default=17, type=int,
                    help='Number of output channel')
parser.add_argument('--pre_resnet', default=True, dest='pre_resnet',
                    help='Use pretrained resnet', action='store_true')
parser.add_argument('--dtype', default='float32', type=str,
                    help='Model dtype')
parser.add_argument('--use_pretrained_base', default=True, dest='use_pretrained_base',
                    help='Use pretrained base', action='store_true')
parser.add_argument('--det_model', default='frcnn', type=str,
                    help='Det model name')
parser.add_argument('--syncbn', default=False, dest='syncbn',
                    help='Use Sync BN', action='store_true')

"----------------------------- Data options -----------------------------"
parser.add_argument('--inputResH', default=256, type=int,
                    help='Input image height')
parser.add_argument('--inputResW', default=192, type=int,
                    help='Input image width')
parser.add_argument('--scale', default=0.3, type=float,
                    help='Degree of scale augmentation')
parser.add_argument('--rotate', default=40, type=float,
                    help='Degree of rotation augmentation')
parser.add_argument('--hmGauss', default=1, type=int,
                    help='Heatmap gaussian size')

"----------------------------- Log options -----------------------------"
parser.add_argument('--logging-file', type=str, default='training.log',
                    help='name of training log file')
parser.add_argument('--board', default=True, dest='board',
                    help='Logging with tensorboard', action='store_true')
parser.add_argument('--visdom', default=False, dest='visdom',
                    help='Visualize with visdom', action='store_true')
parser.add_argument('--map', default=True, dest='map',
                    help='Evaluate mAP per epoch', action='store_true')

"----------------------------- Detection options -----------------------------"
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--mode', dest='mode',
                    help='detection mode, fast/normal/accurate', default="normal")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--inp_dim', dest='inp_dim', type=str, default='608',
                    help='inpdim')
parser.add_argument('--conf', dest='confidence', type=float, default=0.1,
                    help='bounding box confidence threshold')
parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.6,
                    help='bounding box nms threshold')
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--detbatch', type=int, default=1,
                    help='detection batch size')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size')

opt = parser.parse_args()

opt.outputResH = opt.inputResH // 4
opt.outputResW = opt.inputResW // 4

if not os.path.exists("./exp/{}".format(opt.dataset)):
    os.mkdir("./exp/{}".format(opt.dataset))

filehandler = logging.FileHandler(
    './exp/{}/{}-{}'.format(opt.dataset, opt.expID, opt.logging_file))
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info('*************************')
logger.info(opt)
logger.info('*************************')


def epochInfo(self, set, idx, loss, acc):
    if loss is not None:
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))
    else:
        self.info('{set}-{idx:d} epoch | acc:{acc:.4f}'.format(
            set=set,
            idx=idx,
            acc=acc
        ))


logger.epochInfo = MethodType(epochInfo, logger)
