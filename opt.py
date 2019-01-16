import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--dataset', default='coco', type=str,
                    help='Dataset choice: mpii | coco')
parser.add_argument('--nThreads', default=30, type=int,
                    help='Number of data loading threads')
parser.add_argument('--debug', default=False, type=bool,
                    help='Print the debug information')
parser.add_argument('--snapshot', default=1, type=int,
                    help='How often to take a snapshot of the model (0 = never)')

"----------------------------- AlphaPose options -----------------------------"
parser.add_argument('--addDPG', default=False, type=bool,
                    help='Train with data augmentation')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')

"----------------------------- Model options -----------------------------"
parser.add_argument('--netType', default='hgPRM', type=str,
                    help='Options: hgPRM | resnext')
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
parser.add_argument('--Continue', default=False, type=bool,
                    help='Pick up where an experiment left off')
parser.add_argument('--nFeats', default=256, type=int,
                    help='Number of features in the hourglass')
parser.add_argument('--nClasses', default=33, type=int,
                    help='Number of output channel')
parser.add_argument('--nStack', default=4, type=int,
                    help='Number of hourglasses to stack')

"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--fast_inference', default=True, type=bool,
                    help='Fast inference')
parser.add_argument('--use_pyranet', default=True, type=bool,
                    help='use pyranet')

"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--LR', default=2.5e-4, type=float,
                    help='Learning rate')
parser.add_argument('--momentum', default=0, type=float,
                    help='Momentum')
parser.add_argument('--weightDecay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--crit', default='MSE', type=str,
                    help='Criterion type')
parser.add_argument('--optMethod', default='rmsprop', type=str,
                    help='Optimization method: rmsprop | sgd | nag | adadelta')


"----------------------------- Training options -----------------------------"
parser.add_argument('--nEpochs', default=50, type=int,
                    help='Number of hourglasses to stack')
parser.add_argument('--epoch', default=0, type=int,
                    help='Current epoch')
parser.add_argument('--trainBatch', default=40, type=int,
                    help='Train-batch size')
parser.add_argument('--validBatch', default=20, type=int,
                    help='Valid-batch size')
parser.add_argument('--trainIters', default=0, type=int,
                    help='Total train iters')
parser.add_argument('--valIters', default=0, type=int,
                    help='Total valid iters')
parser.add_argument('--init', default=None, type=str,
                    help='Initialization')

"----------------------------- Data options -----------------------------"
parser.add_argument('--inputResH', default=320, type=int,
                    help='Input image height')
parser.add_argument('--inputResW', default=256, type=int,
                    help='Input image width')
parser.add_argument('--outputResH', default=80, type=int,
                    help='Output heatmap height')
parser.add_argument('--outputResW', default=64, type=int,
                    help='Output heatmap width')
parser.add_argument('--scale', default=0.25, type=float,
                    help='Degree of scale augmentation')
parser.add_argument('--rotate', default=30, type=float,
                    help='Degree of rotation augmentation')
parser.add_argument('--hmGauss', default=1, type=int,
                    help='Heatmap gaussian size')

"----------------------------- PyraNet options -----------------------------"
parser.add_argument('--baseWidth', default=9, type=int,
                    help='Heatmap gaussian size')
parser.add_argument('--cardinality', default=5, type=int,
                    help='Heatmap gaussian size')
parser.add_argument('--nResidual', default=1, type=int,
                    help='Number of residual modules at each location in the pyranet')

"----------------------------- Distribution options -----------------------------"
parser.add_argument('--dist', dest='dist', type=int, default=1,
                    help='distributed training or not')
parser.add_argument('--backend', dest='backend', type=str, default='gloo',
                    help='backend for distributed training')
parser.add_argument('--port', dest='port',
                    help='port of server')

"----------------------------- Detection options -----------------------------"
parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                    default='res152')
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
parser.add_argument('--conf', dest='confidence', type=float, default=0.05,
                    help='bounding box confidence threshold')
parser.add_argument('--nms', dest='nms_thesh', type=float, default=0.6,
                    help='bounding box nms threshold')
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--matching', default=False, action='store_true',
                    help='use best matching')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--detbatch', type=int, default=1,
                    help='detection batch size')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size')

"----------------------------- Video options -----------------------------"
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=str,
                    help='webcam number', default='0')
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
opt = parser.parse_args()

opt.num_classes = 80
