import numpy as np
import torch
def track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores):
    hm = hm.cpu().data.numpy()
    online_targets = tracker.update(orig_img,inps,boxes,hm,cropped_boxes,im_name,scores,_debug=False)
    new_boxes,new_scores,new_ids,new_crop = [],[],[],[]
    new_hm = np.zeros(shape=(1,17,64,48))
    for t in online_targets:
        tlbr = t.tlbr
        tid = t.track_id
        thm = np.expand_dims(t.pose,axis=0)
        tcrop = t.crop_box
        tscore = t.detscore
        new_boxes.append(tlbr)
        new_crop.append(tcrop)
        new_hm = np.concatenate((new_hm,thm),axis=0)
        new_ids.append(tid)
        new_scores.append(tscore)
    new_hm = new_hm[1:]
    new_hm = torch.from_numpy(new_hm).to(args.device)
    return new_boxes,new_scores,new_ids,new_hm,new_crop
