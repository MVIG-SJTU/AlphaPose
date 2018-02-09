import h5py
import numpy as np
import sys
import mpii

keys = ['index','person','imgname','center','scale','part','visible','normalize','torsoangle','multi','istrain']
annot = {k:[] for k in keys}

# Set up index reference for multiperson training
multiRef = np.zeros(mpii.nimages)
trainRef = mpii.annot['img_train'][0][0][0]
allIdxs = np.arange(0,trainRef.shape[0])
with h5py.File('../../data/mpii/annot/multi-idxs.h5','r') as f:
    mTrain = f['train'][:] - 1
    mTest = f['test'][:] - 1
    multiRef[allIdxs[trainRef == 1][mTrain]] = 1
    multiRef[allIdxs[trainRef == 0][mTest]] = 1

# Get image filenames
imgnameRef = mpii.annot['annolist'][0][0][0]['image'][:]

for idx in xrange(mpii.nimages):
    print "\r",idx,
    sys.stdout.flush()

    for person in xrange(mpii.numpeople(idx)):
        c,s = mpii.location(idx,person)
        if not c[0] == -1:
            # Add info to annotation list
            annot['index'] += [idx]
            annot['person'] += [person]
            imgname = np.zeros(16)
            refname = str(imgnameRef[idx][0][0][0][0])
            for i in range(len(refname)): imgname[i] = ord(refname[i])
            annot['imgname'] += [imgname]
            annot['center'] += [c]
            annot['scale'] += [s]
            annot['multi'] += [multiRef[idx]]

            if mpii.istrain(idx) == True:
                # Part annotations and visibility
                coords = np.zeros((16,2))
                vis = np.zeros(16)
                for part in xrange(16):
                   coords[part],vis[part] = mpii.partinfo(idx,person,part)
                annot['part'] += [coords]
                annot['visible'] += [vis]
                annot['normalize'] += [mpii.normalization(idx,person)]
                annot['torsoangle'] += [mpii.torsoangle(idx,person)]
                annot['istrain'] += [1]
            else:
                annot['part'] += [-np.ones((16,2))]
                annot['visible'] += [np.zeros(16)]
                annot['normalize'] += [1]
                annot['torsoangle'] += [0]
                if trainRef[idx] == 0:  # Test image
                    annot['istrain'] += [0]
                else:   # Training image (something missing in annot)
                    annot['istrain'] += [2]

print ""

with h5py.File('mpii-annot.h5','w') as f:
    f.attrs['name'] = 'mpii'
    for k in keys:
        f[k] = np.array(annot[k])
