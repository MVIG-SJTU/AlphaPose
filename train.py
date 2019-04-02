import os

import mxnet as mx
import scipy
from gluoncv import data
from gluoncv.utils import LRScheduler, makedirs
from mxnet import autograd, gluon, nd
from tqdm import tqdm

import visdom
from sppe.logger import board_writing, visdom_writing
from sppe.models.sefastpose import FastPose_SE
from opt import logger, opt
from sppe.predict.p_pose_nms import pose_nms
from tensorboardX import SummaryWriter
from sppe.utils.datasets.coco import Mscoco
from sppe.utils.datasets.fuse import Fuse
from sppe.utils.eval import getmap
from sppe.utils.heatmap_acc import HeatmapAccuracy
from sppe.utils.img import (detector_to_simple_pose, flip, heatmap_to_coord,
                            shuffleLR)

ctx = [mx.gpu(int(i)) for i in opt.gpu_id]


if opt.visdom:
    if 'SLURM_SUBMIT_HOST' in os.environ.keys():
        server_host = os.environ['SLURM_SUBMIT_HOST']
    else:
        server_host = 'http://localhost'
    vis = visdom.Visdom(server=server_host, port=8089)
    win = [None] * 10


def train_batch_fn(batch, ctx):
    inps = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    inp_masks = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
    ori_inps = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)
    return inps, labels, inp_masks, ori_inps


def loadModel(m):
    if opt.load_from_pyt:
        m.load_from_pytorch(opt.loadModel, ctx=ctx)
    elif opt.loadModel:
        m.load_parameters(opt.loadModel, ctx=ctx)
    elif opt.try_loadModel:
        m.try_load_parameters(opt.try_loadModel, ctx=ctx)
    else:
        logger.info('Create new model')

    makedirs("./exp/{}/{}".format(opt.dataset, opt.expID))
    return m


def train(train_model, epoch):
    train_model.hybridize(static_alloc=True, static_shape=True)

    loss_value = 0
    train_loader_desc = tqdm(train_loader, ncols=20)

    for i, batch in enumerate(train_loader_desc):
        inps, labels, inp_masks, ori_inps = train_batch_fn(batch, ctx)
        with autograd.record():
            outputs = [train_model(X.astype(opt.dtype, copy=False)) for X in inps]
            loss = [nd.cast(2 * criterion(nd.cast(yhat, opt.dtype), y, w), opt.dtype)
                    for yhat, y, w in zip(outputs, labels, inp_masks)]
        for l in loss:
            l.backward()

        lr_scheduler.update(i, epoch)
        trainer.step(train_batch)

        metric.update(labels, outputs)
        loss_value += sum([l.mean().asscalar() for l in loss]) / num_gpu

        opt.trainIters += 1
        _, acc = metric.get()

        # Tensorboard
        if opt.board:
            board_writing(writer, loss_value / (i + 1), acc, opt.trainIters, 'Train')

        # TQDM
        train_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f} | lr: {lr:.6f}'.format(
                loss=loss_value / (i + 1),
                acc=acc * 100,
                lr=trainer.learning_rate)
        )

        # Visdom
        if opt.visdom and i % 10 == 0:
            global win
            win = visdom_writing(vis, outputs, labels, inputs=inps, ori_inputs=ori_inps, win=win)

    train_loader_desc.close()
    _, acc = metric.get()

    return loss_value / (i + 1), acc


def valid_acc(val_model, epoch):
    val_metric = HeatmapAccuracy()
    val_metric.reset()

    # loss_value = 0
    val_loader_desc = tqdm(val_loader, ncols=20)

    for i, batch in enumerate(val_loader_desc):
        inps, labels, inp_masks, ori_inps = train_batch_fn(batch, ctx)

        outputs = [val_model(X.astype(opt.dtype, copy=False)) for X in inps]

        if len(outputs) > 1:
            outputs_stack = nd.concat(*[out.as_in_context(mx.cpu()) for out in outputs], dim=0)
            labels_stack = nd.concat(*[label.as_in_context(mx.cpu()) for label in labels], dim=0)
            ori_inps_stack = nd.concat(*[ori_inp.as_in_context(mx.cpu()) for ori_inp in ori_inps], dim=0)
        else:
            outputs_stack = outputs[0].as_in_context(mx.cpu())
            labels_stack = labels[0].as_in_context(mx.cpu())
            ori_inps_stack = ori_inps[0].as_in_context(mx.cpu())

        opt.valIters += 1

        val_metric.update(labels_stack, outputs_stack)
        _, acc = val_metric.get()

        # Tensorboard
        if opt.board:
            board_writing(writer, None, acc, opt.valIters, 'Valid')

        # TQDM
        val_loader_desc.set_description(
            'acc: {acc:.2f}'.format(
                acc=acc * 100)
        )

        # Visdom
        if opt.visdom and i % 10 == 0:
            global win
            win = visdom_writing(
                vis, outputs_stack, labels_stack, inputs=inps, ori_inputs=ori_inps_stack, win=win)

    val_loader_desc.close()
    _, acc = val_metric.get()

    return acc


def valid_map(val_model):

    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    from gluoncv import model_zoo
    if opt.det_model == 'yolo':
        detector = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, ctx=mx.gpu(0), root='./exp/pretrained')
        detector.set_nms(nms_thresh=0.45, nms_topk=-1)
        detector.reset_class(["person"], reuse_weights=['person'])
        detector.hybridize()

        load_test = data.transforms.presets.yolo.load_test
    elif opt.det_model == 'frcnn':
        detector = model_zoo.get_model(
            'faster_rcnn_resnet101_v1d_coco', pretrained=True, ctx=mx.gpu(0), root='./exp/pretrained')
        detector.set_nms(nms_thresh=0.6, nms_topk=-1)
        detector.reset_class(["person"], reuse_weights=['person'])
        detector.hybridize()

        load_test = data.transforms.presets.rcnn.load_test

    prefix = './data/coco/images'
    list_file = './data/coco/list-coco-minival500.txt'
    img_names = [img_name.strip('\n') for img_name in open(list_file, 'r').readlines()]

    val_model.collect_params().reset_ctx(mx.gpu(1))

    final_results = []

    for img_name in tqdm(img_names):
        file_name = os.path.join(prefix, img_name)
        # x, img = data.transforms.presets.ssd.load_test(file_name, short=512)
        x, det_img = load_test(file_name, short=800, max_size=1300)
        img = scipy.misc.imread(file_name, mode='RGB')
        # Perform Human Detection
        h_scale = det_img.shape[0] / img.shape[0]
        w_scale = det_img.shape[1] / img.shape[1]

        class_IDs, scores, bounding_boxs = detector(x.as_in_context(mx.gpu(0)))
        pose_input, upscale_bbox, scores, class_IDs, bounding_boxs = detector_to_simple_pose(
            img, class_IDs, scores, bounding_boxs, rescale=(h_scale, w_scale))

        # Apply SPPE
        if pose_input is None:
            continue

        predicted_heatmap = val_model(pose_input.as_in_context(mx.gpu(1)))
        # Flip
        flip_pose_input = flip(pose_input).as_in_context(mx.gpu(1))
        flip_predicted_heatmap = val_model(flip_pose_input)
        flip_predicted_heatmap = shuffleLR(flip(flip_predicted_heatmap), val_loader._dataset)

        predicted_heatmap = (predicted_heatmap + flip_predicted_heatmap) / 2
        predicted_heatmap = predicted_heatmap[:, :17, :, :]

        pred_coords, confidence = heatmap_to_coord(predicted_heatmap.as_in_context(mx.cpu()), upscale_bbox)

        pred_coords, confidence, bounding_boxs = pose_nms(pred_coords, confidence, upscale_bbox, scores, bounding_boxs)

        if pred_coords is None:
            continue
        preds = nd.concat(pred_coords, confidence, dim=2)

        # Post Processing
        num_pred = preds.shape[0]
        for n in range(num_pred):
            # pro_scores = nd.sum(confidence[n]).asscalar() / 17 + nd.max(confidence[n]).asscalar() + scores[n][0]
            pro_scores = scores[n][0]

            pred = preds[n].asnumpy().reshape(17 * 3).tolist()
            if isinstance(bounding_boxs[n], list):
                bbox = bounding_boxs[n].copy()
            else:
                bbox = bounding_boxs[n].asnumpy().tolist().copy()

            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            result = {}
            result['image_id'] = int(img_name.split('/')[-1].split('.')[0].split('_')[-1])
            result['category_id'] = 1
            result['bbox'] = bbox
            result['keypoints'] = pred
            result['score'] = float(pro_scores)

            final_results.append(result)

    mAP_kp, mAP_det = getmap(final_results, file_name=f'{opt.dataset}-{opt.expID}_pose_results.json')

    logger.info(f'mAP_kp: {mAP_kp}, mAP_det: {mAP_det}')

    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '2'


num_gpu = len(opt.gpu_id)

train_batch = opt.trainBatch * num_gpu
valid_batch = opt.validBatch
num_workers = opt.nThreads
kwargs = {}

if opt.syncbn:
    kwargs['num_devices'] = len(opt.gpu_id)

train_model = FastPose_SE(ctx=ctx, **kwargs)
train_model.cast(opt.dtype)

# Prepare dataset
if opt.dataset == 'coco':
    train_dataset = Mscoco(train=True)
    val_dataset = Mscoco(train=False)
elif opt.dataset == 'fuse':
    train_dataset = Fuse(train=True)
    val_dataset = Fuse(train=False)

train_loader = gluon.data.DataLoader(
    train_dataset, batch_size=train_batch, shuffle=True, num_workers=num_workers, last_batch='discard')

val_loader = gluon.data.DataLoader(
    val_dataset, batch_size=valid_batch, shuffle=False, num_workers=num_workers, last_batch='discard')

num_training_samples = len(train_dataset)
num_valid_samples = len(val_dataset)

lr_decay = opt.lr_decay
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]

num_batches = num_training_samples // train_batch
lr_scheduler = LRScheduler(mode='step', baselr=opt.LR,
                           niters=num_batches, nepochs=opt.nEpochs,
                           step=lr_decay_epoch, step_factor=lr_decay, power=2)

optimizer = opt.optMethod
optimizer_params = {'wd': 0.0, 'lr_scheduler': lr_scheduler}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True

if opt.optMethod == 'rmsprop':
    optimizer_params['gamma1'] = 0.99
    optimizer_params['epsilon'] = 1e-16

train_model = loadModel(train_model)

if not opt.loadModel and not opt.try_loadModel:
    train_model.initialize(mx.init.MSRAPrelu(), ctx=ctx, force_reinit=False)
    # train_model.reload_base()
    # train_model.reset_ctx(ctx)

trainer = gluon.Trainer(train_model.collect_params(), optimizer, optimizer_params)
criterion = gluon.loss.L2Loss()
metric = HeatmapAccuracy()
writer = SummaryWriter(
    '.tensorboard/{}/{}'.format(opt.dataset, opt.expID))

train_model.hybridize(static_alloc=True, static_shape=True)

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '2'

for epoch in range(opt.nEpochs):
    opt.epoch = epoch
    metric.reset()

    # valid_map(train_model)

    logger.info('############# Starting Epoch {} #############'.format(opt.epoch))
    # Training
    train_model.collect_params().reset_ctx(ctx)

    loss, acc = train(train_model, epoch)
    logger.epochInfo('Train', opt.epoch, loss, acc)

    if epoch % opt.snapshot == 0:
        train_model.save_parameters('./exp/{}/{}/model_{}.params'.format(opt.dataset, opt.expID, opt.epoch))
        trainer.save_states('./exp/{}/{}/model_{}.states'.format(opt.dataset, opt.expID, opt.epoch))

        # Validation
        val_model = train_model
        # val_model = create_model(opt.version, ctx)
        # val_model.cast(opt.dtype)
        # val_model.load_parameters(('./exp/{}/{}/model_{}.params'.format(opt.dataset, opt.expID, opt.epoch)), ctx=ctx)
        acc = valid_acc(val_model, epoch)
        logger.epochInfo('Valid', opt.epoch, None, acc)
    if acc > 0.7:
        valid_map(train_model)

writer.close()
