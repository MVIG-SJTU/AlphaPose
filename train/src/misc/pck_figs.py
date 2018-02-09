import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import h5py

def setuppdjplot(ax, i):
    # Configuration of ticks in plots
    major_ticks_y = np.arange(0,1.01,.2)
    minor_ticks_y = np.arange(0,1.01,.1)
    major_ticks_x = np.arange(0,.21,.1)
    minor_ticks_x = np.arange(0,.21,.05)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    for tick in ax.yaxis.get_major_ticks():
        # tick.label.set_fontsize(8)
        if i == 0:
            tick.label.set_visible(False)
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(8)
    ax.grid()
    ax.grid(which='minor', alpha=0.5)

def plottraintest(ax, train_log, test_log, title='Loss'):
    idx = [0,2]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    t = (np.arange(len(train_log[1:,idx[0]])) + 1)
    ax.plot(t, train_log[1:,idx[0]], label='Train', color='k')
    t = (np.arange(len(test_log[1:,idx[1]])) + 1)
    ax.plot(t, test_log[1:,idx[1]], label='Test', color='r')
    # ax.set_ylim(0,1)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Training/Test %s'%title, fontsize=10)

def loadpreds(predfile, obs):
    with h5py.File(predfile, 'r') as f:
        preds = np.array(f['preds_tf'])
        dist_key = 'dist_'
        if obs: dist_key += 'o'
        dists = np.array(f[dist_key])
    return preds, dists

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

def pdjdata(dataset, dists, partidx, rng=None, filt=None):
    # Return data for creating a PDJ plot
    # Returns the average curve for the parts provided

    if rng is None:
        # If no range is provided use the default ranges for flic and mpii
        if dataset == 'flic':
            rng = [0, .21, .01]
        elif dataset == 'mpii':
            rng = [0, .51, .01]

    t = np.arange(rng[0],rng[1],rng[2])
    pdj = np.zeros(len(t))

    for i in xrange(len(t)):
        for j in partidx:
            pdj[i] += getaccuracy(dists[:, j], t[i], filt=filt)

    pdj /= len(partidx) # Average across all chosen parts
    return pdj, t

# =============================================================================
# MPII Figures
# =============================================================================

dirnames = ['carreira15arxiv','pishchulin15arxiv',
            'tompson14nips','tompson15cvpr',
            'wei16arxiv','our_model']

results = []

for d in dirnames:
    results += [loadmat('mpii_results/'+d+'/pckAll.mat')]
'''
plt.plot(results[0]['range'].T,results[0]['pck'][:,0])
plt.plot(results[0]['range'].T,results[1]['pck'][:,0])
plt.plot(results[0]['range'].T,results[2]['pck'][:,0])
plt.plot(results[0]['range'].T,results[3]['pck'][:,0])
plt.plot(results[0]['range'].T,results[4]['pck'][:,0])
plt.plot(results[0]['range'].T,results[5]['pck'][:,0])
plt.show()
'''

# =============================================================================
# FLIC Figures
# =============================================================================

# flic_wri = loadmat('flic_results_wrist')
# flic_wri = flic_wri['new_data']
# flic_elb = loadmat('flic_results_elbow')
# flic_elb = flic_elb['new_data']
# nyu_dists = np.load('/home/-/posenet/data/flic/ref/nyu_dists_flic_obs.npy')
# _,our_dists = loadpreds('/home/-/posenet/exp/flic/base/preds.h5',True)

# # Plot elbow results
# nyu_elb,t = pdjdata('flic',nyu_dists,[1,4])
# our_elb,_ = pdjdata('flic',our_dists,[1,4])
# f = plt.figure(facecolor='w')
# ax = f.add_subplot(1,2,2)
# lines = ax.plot(t,our_elb)
# plt.setp(lines,linewidth=2)
# lines = ax.plot(t,nyu_elb)
# plt.setp(lines,linewidth=2)
# lines = ax.plot(t,flic_elb[[0,2],:21].T)
# setuppdjplot(ax, 0)
# ax.set_title('Elbow',fontsize=24)
# ax.set_ylabel('Detection Rate (%)',fontsize=22)
# ax.set_xlabel('Normalized Distance',fontsize=22)
# slabels = ['Ours','Tompson et al.','Chen et al.','Toshev et al.']
# plt.setp(lines,linewidth=2)
# ax.legend(loc=4,labels=labels, fontsize=16)

# # Plot elbow results
# nyu_wri,t = pdjdata('flic',nyu_dists,[2,5])
# our_wri,_ = pdjdata('flic',our_dists,[2,5])
# ax = f.add_subplot(1,2,1)
# lines = ax.plot(t,our_wri)
# plt.setp(lines,linewidth=2)
# lines = ax.plot(t,nyu_wri)
# plt.setp(lines,linewidth=2)
# lines = ax.plot(t,flic_wri[[0,2],:21].T)
# setuppdjplot(ax, 1)
# ax.set_ylabel('Detection Rate (%)',fontsize=22)
# ax.set_xlabel('Normalized Distance',fontsize=22)
# ax.set_title('Wrist',fontsize=24)
# plt.setp(lines,linewidth=2)

# ax2 = f.add_subplot(1,1,1)
# ax2.axis('off')
# ax2.set_xlabel('Normalized Distance')
# t = f.suptitle('FLIC Results',fontsize=32)
# plt.show()


# =============================================================================
# Loss comparisons
# =============================================================================

exps = ['307-DR', '304-hg-D', 'hg-1_s2_b1', 'F28-hg-I-2', '304-hg-IA']

logs = []
rounds = []
for exp in exps:
    log = np.loadtxt('log_data/mpii/'+exp+'/test.log',skiprows=1)
    rounds += [log[:,1]]
    logs += [log[:,0]]

f = plt.figure(facecolor='w')
ax = f.add_subplot(111)
f.suptitle('Validation Accuracy Across Training',fontsize=32)
ax.set_ylabel('Average Accuracy (%)',fontsize=24)
ax.set_xlabel('Training Iterations (x2000)',fontsize=24)
for i in xrange(len(exps)):
    ln = ax.plot(rounds[i],logs[i])
    plt.setp(ln,linewidth=1.5)

labels = ['HG','HG-Int','HG-Stacked','HG-Stacked-Int','HG-Stacked-Add']
ax.legend(loc=4,labels=labels,fontsize=16)
ax.set_xlim(0,100)
ax.set_ylim(0,1)

plt.show()