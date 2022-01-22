"""Validation script."""
import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import sys

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import (flip, flip_heatmap,
                                        get_func_heatmap_to_coord)


parser = argparse.ArgumentParser(description='AlphaPose Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    required=True,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    default='0',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    default=32,
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--oks-nms',
                    default=False,
                    dest='oks_nms',
                    help='use oks nms',
                    action='store_true')
parser.add_argument('--ppose-nms',
                    default=False,
                    dest='ppose_nms',
                    help='use pPose nms, recommended',
                    action='store_true')

opt = parser.parse_args()
cfg = update_config(opt.cfg)

gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [gpus[0]]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")


def validate(m, heatmap_to_coord, batch_size=20):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    eval_joints = det_dataset.EVAL_JOINTS

    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')

    halpe = (cfg.DATA_PRESET.NUM_JOINTS == 133) or (cfg.DATA_PRESET.NUM_JOINTS == 136)

    for inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds in tqdm(det_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]
        else:
            inps = inps.cuda()
        output = m(inps)
        if opt.flip_test:
            if isinstance(inps, list):
                inps_flip = [flip(inp).cuda() for inp in inps]
            else:
                inps_flip = flip(inps).cuda()
            output_flip = flip_heatmap(m(inps_flip), det_dataset.joint_pairs, shift=True)
            pred_flip = output_flip[:, eval_joints, :, :]
        else:
            output_flip = None
            pred_flip = None

        pred = output
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        if output.size()[1] == 68:
            face_hand_num = 42
        else:
            face_hand_num = 110

        for i in range(output.shape[0]):
            bbox = crop_bboxes[i].tolist()
            if combined_loss:
                pose_coords_body_foot, pose_scores_body_foot = heatmap_to_coord[0](
                    pred[i][det_dataset.EVAL_JOINTS[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type, 
                    hms_flip=pred_flip[i][det_dataset.EVAL_JOINTS[:-face_hand_num]] if pred_flip is not None else None)
                pose_coords_face_hand, pose_scores_face_hand = heatmap_to_coord[1](
                    pred[i][det_dataset.EVAL_JOINTS[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type, 
                    hms_flip=pred_flip[i][det_dataset.EVAL_JOINTS[-face_hand_num:]] if pred_flip is not None else None)
                pose_coords = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_scores = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coords, pose_scores = heatmap_to_coord(
                    pred[i][det_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type, 
                    hms_flip=pred_flip[i][det_dataset.EVAL_JOINTS] if pred_flip is not None else None)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i, 0].tolist()
            data['image_id'] = int(img_ids[i])
            data['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            data['score'] = float(scores[i] + np.mean(pose_scores) + 1.25 * np.max(pose_scores))
            # data['score'] = float(scores[i])
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    if opt.ppose_nms:
        from alphapose.utils.pPose_nms import ppose_nms_validate_preprocess, pose_nms, write_json
        final_result = []
        tmp_data = ppose_nms_validate_preprocess(kpt_json)
        for key in tmp_data:
            boxes, scores, ids, preds_img, preds_scores = tmp_data[key]
            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, 0, cfg.LOSS.get('TYPE') == 'MSELoss')

            _result = []
            for k in range(len(scores)):
                _result.append(
                    {
                        'keypoints':preds_img[k],
                        'kp_score':preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx':ids[k],
                        'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]]
                    }
                ) 
            im_name = str(key).zfill(12) + '.jpg'
            result = {
                'imgname': im_name,
                'result': _result
            }
            final_result.append(result)

        write_json(final_result, './exp/json/', form='coco', for_eval=True, outputfile='validate_rcnn_kpt.json')
    else:
        if opt.oks_nms:
            from alphapose.utils.pPose_nms import oks_pose_nms
            kpt_json = oks_pose_nms(kpt_json)

        with open('./exp/json/validate_rcnn_kpt.json', 'w') as fid:
            json.dump(kpt_json, fid)

    sysout = sys.stdout
    res = evaluate_mAP('./exp/json/validate_rcnn_kpt.json', ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.TEST.ROOT, cfg.DATASET.TEST.ANN), halpe=halpe)
    sys.stdout = sysout
    return res


def validate_gt(m, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')

    halpe = (cfg.DATA_PRESET.NUM_JOINTS == 133) or (cfg.DATA_PRESET.NUM_JOINTS == 136)

    for inps, labels, label_masks, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]
        else:
            inps = inps.cuda()
        output = m(inps)
        if opt.flip_test:
            if isinstance(inps, list):
                inps_flip = [flip(inp).cuda() for inp in inps]
            else:
                inps_flip = flip(inps).cuda()
            output_flip = flip_heatmap(m(inps_flip), gt_val_dataset.joint_pairs, shift=True)
            pred_flip = output_flip[:, eval_joints, :, :]
        else:
            output_flip = None
            pred_flip = None

        pred = output
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        if output.size()[1] == 68:
            face_hand_num = 42
        else:
            face_hand_num = 110

        for i in range(output.shape[0]):
            bbox = bboxes[i].tolist()
            if combined_loss:
                pose_coords_body_foot, pose_scores_body_foot = heatmap_to_coord[0](
                    pred[i][gt_val_dataset.EVAL_JOINTS[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type, 
                    hms_flip=pred_flip[i][gt_val_dataset.EVAL_JOINTS[:-face_hand_num]] if pred_flip is not None else None)
                pose_coords_face_hand, pose_scores_face_hand = heatmap_to_coord[1](
                    pred[i][gt_val_dataset.EVAL_JOINTS[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type, 
                    hms_flip=pred_flip[i][gt_val_dataset.EVAL_JOINTS[-face_hand_num:]] if pred_flip is not None else None)
                pose_coords = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_scores = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coords, pose_scores = heatmap_to_coord(
                    pred[i][gt_val_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type, 
                    hms_flip=pred_flip[i][gt_val_dataset.EVAL_JOINTS] if pred_flip is not None else None)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + 1.25 * np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    sysout = sys.stdout
    with open('./exp/json/validate_gt_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP('./exp/json/validate_gt_kpt.json', ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN), halpe=halpe)
    sys.stdout = sysout
    return res


if __name__ == "__main__":
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint))

    m = torch.nn.DataParallel(m, device_ids=gpus).cuda()
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    with torch.no_grad():
        gt_AP = validate_gt(m, cfg, heatmap_to_coord, opt.batch)
        detbox_AP = validate(m, heatmap_to_coord, opt.batch)
    print('##### gt box: {} mAP | det box: {} mAP #####'.format(gt_AP, detbox_AP))