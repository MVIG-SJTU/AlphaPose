# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:58:09 2016

@author: benjamin
"""

import json
import numpy as np
from six.moves import xrange
import h5py
import os
import argparse

def write_nms_json(outputpath, sep, form):
    os.chdir(os.path.join(outputpath,'POSE'))
    pred_file=[line.rstrip('\n').rstrip(' ') for line in open("pred.txt")]
    score_file=[line.rstrip('\n').rstrip(' ') for line in open("scores.txt")]
    proposal_scores = np.loadtxt("scores-proposals.txt", ndmin=1)
     
    results = []
    bbox_cnt = 0
    for i in xrange(len(pred_file)):
        keypoints = []
        score = []
        score2 = []
        pred_coordi = pred_file[i].split('\t')
        pred_score = score_file[i].split('\t')
        result = {}
        result['image_id'] = pred_coordi[0]
        result['category_id'] = 1;
        for n in xrange(16):
            keypoints.append(int(pred_coordi[2*n+1])); 
            keypoints.append(int(pred_coordi[2*n+2]));
            keypoints.append(float(pred_score[n+1]));
            score.append(float(pred_score[n+1]))
            if float(pred_score[n+1]) > 0.3:
                score2.append(float(pred_score[n + 1]))
        if len(score2) == 0:
            score2 = [0.3]
        bbox_cnt += 1
        result['keypoints'] = keypoints
        result['score'] = 1.0*np.mean(score) + 0.5 * proposal_scores[i] + 1.25 * np.max(score) + np.mean(score2)
        results.append(result)
    results_forvis = {}
    last_image_name = ' '
    for i in xrange(len(results)):
        imgpath = results[i]['image_id']
        if last_image_name != imgpath:
            results_forvis[imgpath] = []
            results_forvis[imgpath].append({'keypoints':results[i]['keypoints'],'scores':results[i]['score']})
        else:
            results_forvis[imgpath].append({'keypoints':results[i]['keypoints'],'scores':results[i]['score']})
        last_image_name = imgpath
    with open("alpha-pose-results-forvis.json",'w') as json_file:
            json_file.write(json.dumps(results_forvis))
    if form == 'default':    
        with open("alpha-pose-results.json",'w') as json_file:
            json_file.write(json.dumps(results))
        if sep == 'true':
            if not os.path.exists('sep-json'):
                os.mkdir('sep-json')
            result2={}
            for item in results:
                if item['image_id'] not in result2.keys():
                    result2[item['image_id']]=[]
                result2[item['image_id']].append(item)
            for name in result2.keys():
                with open('sep-json/'+("%s"%name.split('.')[0]+'.json').split('/')[-1],'w') as json_file:
                    json_file.write(json.dumps(result2[name]))
    elif form == 'cmu': # the form of CMU-Pose/OpenPose
        result3={}
        for item in results:
            if item['image_id'] not in result3.keys():
                result3[item['image_id']]={}
                result3[item['image_id']]['version']=0.1
                result3[item['image_id']]['bodies']=[]
            tmp={'joints':[]}
            indexarr=[27,24,36,33,30,39,42,45,6,3,0,9,12,15,21]
            for i in indexarr:
                tmp['joints'].append(item['keypoints'][i])
                tmp['joints'].append(item['keypoints'][i+1])
                tmp['joints'].append(item['keypoints'][i+2])
            result3[item['image_id']]['bodies'].append(tmp)
        with open("alpha-pose-results.json",'w') as json_file:
            json_file.write(json.dumps(result3))
        if sep == 'true':   
            if not os.path.exists('sep-json'):
                os.mkdir('sep-json')
            for name in result3.keys():
                with open('sep-json/'+("%s"%name.split('.')[0]+'.json').split('/')[-1],'w') as json_file:
                    json_file.write(json.dumps(result3[name]))
    else:
        print("format must be either 'coco' or 'cmu'")

def test_parametric_pose_NMS_json(delta1,delta2,mu,gamma,outputpath):
    scoreThreds = 0.3

    #prepare data
    h5file = h5py.File(os.path.join(outputpath,"POSE/test-pose.h5"), 'r')
    preds = np.array(h5file['preds'])
    scores = np.array(h5file['scores'])  
    scores[scores==0] = 1e-5  
    indexs = [line.rstrip(' ').rstrip('\r').rstrip('\n') for line in open(os.path.join(outputpath,"BBOX/index.txt"))]
    scores_proposals = np.loadtxt(os.path.join(outputpath,"BBOX/score-proposals.txt"), ndmin=1)

    #get bounding box sizes    
    bbox_file = h5py.File(os.path.join(outputpath,"BBOX/test-bbox.h5"),'r')
    xmax=np.array(bbox_file['xmax']); xmin=bbox_file['xmin']; ymax=np.array(bbox_file['ymax']); ymin=bbox_file['ymin']
    widths=xmax-xmin; heights=ymax-ymin;
    alpha = 0.1
    Sizes=alpha*np.maximum(widths,heights)
    area = widths*heights
    
    #set the corresponding dir
    if (os.path.exists(os.path.join(outputpath,'POSE')) == False):
        os.mkdir(os.path.join(outputpath,'POSE'))
    os.chdir(os.path.join(outputpath,'POSE'))
    
    NMS_preds = open("pred.txt",'w')
    NMS_scores = open("scores.txt",'w')
    proposal_scores = open("scores-proposals.txt",'w')
    NMS_index = open("index.txt",'w')
    num_human = 0
    
    #loop through every image
    for i in xrange(len(indexs)):
        index = indexs[i].split(' '); 
        img_name = index[0]; start = int(index[1])-1; end = int(index[2])-1;
        
        #initialize scores and preds coordinates
        img_preds = preds[start:end+1]; img_scores = np.mean(scores[start:end+1],axis = 1)
        img_ids = np.arange(end-start+1); ref_dists = Sizes[start:end+1];keypoint_scores = scores[start:end+1];
        pro_score = scores_proposals[start:end+1]
        
        #do NMS by parametric
        pick = []
        merge_ids = []
        while(img_scores.size != 0):
            
            #pick the one with highest score
            pick_id = np.argmax(img_scores)  
            pick.append(img_ids[pick_id])
            
            #get numbers of match keypoints by calling PCK_match 
            ref_dist=ref_dists[img_ids[pick_id]]
            simi = get_parametric_distance(pick_id,img_preds, keypoint_scores,ref_dist, delta1, delta2, mu)
            
            #delete humans who have more than matchThreds keypoints overlap with the seletced human.
            delete_ids = np.arange(img_scores.shape[0])[simi > gamma]
            if (delete_ids.size == 0):
                delete_ids = pick_id
            merge_ids.append(img_ids[delete_ids])
            img_preds = np.delete(img_preds,delete_ids,axis=0); img_scores = np.delete(img_scores, delete_ids)
            img_ids = np.delete(img_ids, delete_ids); keypoint_scores = np.delete(keypoint_scores,delete_ids,axis=0)
        
        #write the NMS result to files
        pick = [Id+start for Id in pick] 
        merge_ids = [Id+start for Id in merge_ids]
        assert len(merge_ids) == len(pick)
        preds_pick = preds[pick]; scores_pick = scores[pick];sizes_pick = Sizes[pick];
        num_pick = 0
        for j in xrange(len(pick)):
            
            #first compute the average score of a person
            ids = np.arange(16)
            max_num = sum(scores_pick[j,ids,0] > scoreThreds)
            if max_num < 3:
                continue
            
            # merge poses
            merge_id = merge_ids[j]  
            score = scores_proposals[pick[j]]
            merge_poses,merge_score = merge_pose(preds_pick[j],preds[merge_id],scores[merge_id],Sizes[pick[j]])
            
            ids = np.arange(16)
            max_num = sum(scores_pick[j,ids,0] > scoreThreds)
            if max_num < 3:
                continue
            
            #add the person to predict
            num_pick += 1
            NMS_preds.write("{}".format(img_name))
            NMS_scores.write("{}".format(img_name))
            proposal_scores.write("{}\n".format(score))
            
            for point_id in xrange(16):
                NMS_preds.write("\t{}\t{}".format(int(merge_poses[point_id,0]),int(merge_poses[point_id,1])))
                NMS_scores.write("\t{}".format(merge_score[point_id]))
            NMS_preds.write("\n")
            NMS_scores.write("\n")
        NMS_index.write("{} {} {}\n".format(img_name, num_human+1, num_human + num_pick))
        num_human += num_pick
        
    NMS_preds.close();NMS_scores.close();NMS_index.close(); proposal_scores.close()
    
def get_parametric_distance(i,all_preds, keypoint_scores,ref_dist, delta1, delta2, mu):
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]
    dist = np.sqrt(np.sum(np.square(pick_preds[np.newaxis,:]-all_preds),axis=2))/ref_dist
    mask = (dist <= 1)
    # defien a keypoints distances
    score_dists = np.zeros([all_preds.shape[0], 16])
    keypoint_scores = np.squeeze(keypoint_scores)
    if (keypoint_scores.ndim == 1) :
        keypoint_scores = keypoint_scores[np.newaxis,:]
    # the predicted scores are repeated up to do boastcast
    pred_scores = np.tile(pred_scores, [1,all_preds.shape[0]]).T
    score_dists[mask] = np.tanh(pred_scores[mask]/delta1)*np.tanh(keypoint_scores[mask]/delta1)
    # if the keypoint isn't inside the bbox, set the distance to be 10
#    dist[dist>1] = 10
    point_dist = np.exp((-1)*dist/delta2)
    final_dist = np.sum(score_dists,axis=1)+mu*np.sum(point_dist,axis=1)
    return final_dist
    
def merge_pose(refer_pose, cluster_preds, cluster_keypoint_scores, ref_dist):
    dist = np.sqrt(np.sum(np.square(refer_pose[np.newaxis,:]-cluster_preds),axis=2))
    # mask is an nx16 matrix
    mask = (dist <= ref_dist)
    final_pose = np.zeros([16,2]); final_scores = np.zeros(16)
    if (cluster_preds.ndim == 2):
        cluster_preds = cluster_preds[np.newaxis,:,:]
        cluster_keypoint_scores = cluster_keypoint_scores[np.newaxis,:]
    if (mask.ndim == 1):
        mask = mask[np.newaxis,:]
    for i in xrange(16):
        cluster_joint_scores = cluster_keypoint_scores[:,i][mask[:,i]]
        
        # pick the corresponding i's matched keyjoint locations and do an weighed sum.
        cluster_joint_location = cluster_preds[:,i,:][np.tile(mask[:,i,np.newaxis],(1,2))].reshape(np.sum(mask[:,i,np.newaxis]),-1)

        # get an normalized score
        normed_scores = cluster_joint_scores / np.sum(cluster_joint_scores)
        # merge poses by a weighted sum
        final_pose[i,0] = np.dot(cluster_joint_location[:,0], normed_scores)
        final_pose[i,1] = np.dot(cluster_joint_location[:,1], normed_scores)
        final_scores[i] = np.max(cluster_joint_scores)
    return final_pose, final_scores
    

def get_result_json(args):
    delta1 = 0.01; mu = 2.08; delta2 = 2.08;
    gamma = 22.48;
    test_parametric_pose_NMS_json(delta1, delta2, mu, gamma,args.outputpath)
    write_nms_json(args.outputpath, args.sep, args.format)
  
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='NMS')
    parser.add_argument('--outputpath',dest='outputpath',help='output-directory')
    parser.add_argument('--sep',dest='sep',help='seperate-json')
    parser.add_argument('--format',dest='format', help='json format, options are default or cmu', default='default')
    args = parser.parse_args()
    return args
                                  
args = parse_args()
get_result_json(args)
