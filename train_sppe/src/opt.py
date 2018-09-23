# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--dataset', default='coco', type=str,
                    help='Dataset choice: mpii | coco')
parser.add_argument('--nThreads', default=30, type=int,
                    help='Number of data loading threads')
parser.add_argument('--snapshot', default=1, type=int,
                    help='How often to take a snapshot of the model (0 = never)')

"----------------------------- AlphaPose options -----------------------------"
parser.add_argument('--addDPG', default=False, dest='addDPG',
                    help='Train with data augmentation', action='store_true')

"----------------------------- Model options -----------------------------"
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
parser.add_argument('--nClasses', default=17, type=int,
                    help='Number of output channel')

"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--LR', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--momentum', default=0, type=float,
                    help='Momentum')
parser.add_argument('--weightDecay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='epsilon')
parser.add_argument('--crit', default='MSE', type=str,
                    help='Criterion type')
parser.add_argument('--optMethod', default='rmsprop', type=str,
                    help='Optimization method: rmsprop | sgd | nag | adadelta')


"----------------------------- Training options -----------------------------"
parser.add_argument('--nEpochs', default=50, type=int,
                    help='Number of hourglasses to stack')
parser.add_argument('--epoch', default=0, type=int,
                    help='Current epoch')
parser.add_argument('--trainBatch', default=128, type=int,
                    help='Train-batch size')
parser.add_argument('--validBatch', default=24, type=int,
                    help='Valid-batch size')
parser.add_argument('--trainIters', default=0, type=int,
                    help='Total train iters')
parser.add_argument('--valIters', default=0, type=int,
                    help='Total valid iters')

"----------------------------- Data options -----------------------------"
parser.add_argument('--inputResH', default=320, type=int,
                    help='Input image height')
parser.add_argument('--inputResW', default=256, type=int,
                    help='Input image width')
parser.add_argument('--outputResH', default=80, type=int,
                    help='Output heatmap height')
parser.add_argument('--outputResW', default=64, type=int,
                    help='Output heatmap width')
parser.add_argument('--scale', default=0.3, type=float,
                    help='Degree of scale augmentation')
parser.add_argument('--rotate', default=40, type=float,
                    help='Degree of rotation augmentation')
parser.add_argument('--hmGauss', default=1, type=int,
                    help='Heatmap gaussian size')


opt = parser.parse_args()
