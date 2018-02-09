
import matplotlib.pyplot as plt
from pypose import ref
import h5py
import numpy as np

a = ref.load('mpii','valid')

'''
predFile = '/home/-/posenet/exp/mpii/hg-I-2/preds.h5'
f = h5py.File(predFile,'r')
p = f['preds_tf']
hms = f['pred_heatmaps']

max_act = np.zeros((2958,16))
mean_act = np.zeros((2958,16))

for i in xrange(2958):
    print i
    for j in xrange(16):
        max_act[i][j] = hms[i][j].max()
        mean_act[i][j] = hms[i][j].mean()

np.save('maxAct.npy',max_act)
np.save('meanAct.npy',mean_act)
'''

maxAct = np.load('maxAct.npy')
meanAct = np.load('meanAct.npy')

thr = np.arange(0.01,.9,.01)
thr2 = np.arange(-0.00001,.015,.0002)

# precision = tp / tp + fp
# recall = tp / tp + fn
partChoice = [1,4]
ptIdx = a['part'][:,partChoice,0] <= 0
mxp = []
mxr = []
mnp = []
mnr = []
track_tp = []
track_fp = []
track_fn = []
track_tn = []
max_acc = 0
for i in xrange(thr2.size):
    maxIdx = maxAct[:,partChoice] <= thr[i]
    tp = (maxIdx * ptIdx).sum()
    fp = (maxIdx * -ptIdx).sum()
    fn = (-maxIdx * ptIdx).sum()
    if tp+fp == 0:
        mxp += [1.]
    else:
        mxp += [(float(tp)/(tp+fp))]
    mxr += [(float(tp)/(tp+fn))]
    meanIdx = meanAct[:,partChoice] < thr2[i]
    tp = (meanIdx * ptIdx).sum()
    fp = (meanIdx * -ptIdx).sum()
    fn = (-meanIdx * ptIdx).sum()
    tn = (-meanIdx * -ptIdx).sum()
    acc = float(tp + tn) / (tp + fp +fn + tn)
    # if acc > max_acc:
    #         print thr2[i],acc
    #         max_acc = acc
    if tp+fp == 0:
        mnp += [1.]
    else:
        mnp += [(float(tp)/(tp+fp))]
    mnr += [(float(tp)/(tp+fn))]
    track_tp += [tp]
    track_fp += [fp]
    track_fn += [fn]
    track_tn += [tn]

f = plt.figure()
ax1 = f.add_subplot(111)
ax1.plot(mxr,mxp,label='Max')
ax1.plot(mnr,mnp,label='Mean')
ax1.set_ylim(0,1)
ax1.set_xlim(0,1)
ax1.set_title('Ankle')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.legend(loc='lower right')
"""
ax2 = f.add_subplot(212)
ax2.plot(track_tp,label='tp')
ax2.plot(track_fp,label='fp')
ax2.plot(track_fn,label='fn')
ax2.plot(track_tn,label='tn')
ax2.plot(mnp,label='p')
ax2.plot(mnr,label='r')
ax2.legend()
"""
plt.show()
