import numpy as np
import ref
import img

# Reference for other predictions
other_preds = {'nyu':{'flic':'nyu_pred', 'mpii':'nyu_pred'}}
def get_path(dataset_name, file_name):
    return ref.posedir + '/data/' + dataset_name + '/ref/' + file_name + '.npy'

# Load ground truth annotations
annot = {'flic':ref.load('flic','test'),
         'mpii':ref.load('mpii','valid'),
         'mpii_train':ref.load('mpii','train'),
         'mpii_test':ref.load('mpii','test')}

def getdists(pred, dotrain=False):
    # Get normalized distances between predictions and ground truth

    # Automatically figures out dataset based on number of parts
    if pred.shape[1] == 11:
        dataset = 'flic'
    elif pred.shape[1] == 16:
        dataset = 'mpii'
    else:
        print "Error: Bad prediction file."
        return 0

    idx_ref = []
    if dotrain:
        idx_ref = np.load(get_path(dataset,'train_idxs'))
        dataset += '_train'
        dists = np.zeros((len(idx_ref),annot[dataset]['part'].shape[1]))
    else:
        dists = np.zeros(annot[dataset]['part'].shape[:2])

    # Loop through samples and parts
    for i in xrange(dists.shape[0]):
        if dotrain:
            idx = idx_ref[i]
        else:
            idx = i
        scale = annot[dataset]['normalize'][idx]
        for j in xrange(dists.shape[1]):
            if annot[dataset]['part'][i,j,0] <= 0:
                dists[i,j] = -1
            else:
                dists[i,j] = np.linalg.norm(annot[dataset]['part'][idx,j] - pred[i,j]) / scale
    return dists

def getaccuracy(arr, thresh, no_zero=True, filt=None):
    # Returns number of elements in arr that fall below the given threshold
    # filt should be a binary array the same size as arr
    if filt is None:
        # If no filter has been provided create entirely true array
        filt = np.array([True for _ in xrange(len(arr))])
    else:
        filt = filt.copy()

    if no_zero:
        filt *= (arr > 0)

    return float(sum(arr[filt] <= thresh)) / filt.sum()

def pdjdata(dataset, dists, partnames=None, rng=None, filt=None):
    # Return data for creating a PDJ plot
    # Returns the average curve for the parts provided

    if partnames is None:
        partnames = ref.parts[dataset]

    if rng is None:
        # If no range is provided use the default ranges for flic and mpii
        if dataset == 'flic':
            rng = [0, .21, .01]
        elif dataset == 'mpii':
            rng = [0, .51, .01]

    t = np.arange(rng[0],rng[1],rng[2])
    pdj = np.zeros(len(t))

    if filt is None or filt.sum() > 0:
        for choice in partnames:
            part_idx = ref.parts[dataset].index(choice)
            for i in xrange(len(t)):
                pdj[i] += getaccuracy(dists[:, part_idx], t[i], filt=filt)

        pdj /= len(partnames) # Average across all chosen parts

    return pdj, t

def transformpreds(dataset, preds, res, rot=False, dotrain=False):
    # Predictions from torch will need to go through a coordinate transformation
    new_preds = np.zeros(preds.shape)
    idx_ref = np.arange(len(new_preds))
    if dotrain:
        idx_ref = np.load(get_path(dataset,'train_idxs'))
        dataset += '_train'
    for i in xrange(preds.shape[0]):
        idx = idx_ref[i]
        c = annot[dataset]['center'][idx]
        s = annot[dataset]['scale'][idx]
        if rot:
            r = annot[dataset]['torsoangle'][idx]
        else:
            r = 0
        for j in xrange(preds.shape[1]):
            new_preds[i,j] = img.transform(preds[i,j]-.5, c, s, res, invert=1, rot=r)
    return new_preds
