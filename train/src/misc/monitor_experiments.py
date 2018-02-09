import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from subprocess import call
import time
from datetime import datetime
import sys

plt.ion()
try:
    experiments_to_show = sys.argv[1].split(',')
except:
    print "Error: No experiments provided"
    exit()

print "Monitoring the following experiments:",
for exp in experiments_to_show: print exp,
print ""

track_multiple = sys.argv[2] == '1'
if track_multiple:
    exp_to_track = experiments_to_show[0]
    print "Tracking all variations of:",exp_to_track
    experiments_to_show.remove(exp_to_track)

def readlog(filepath):
    done_first = False
    arr = None
    with open(filepath,'r') as f:
        for line in f:
            if not done_first:
                done_first = True
            else:
                vals = np.array(map(float,line.split()))
                if arr is None:
                    arr = vals.reshape(1,np.size(vals))
                else:
                    arr = np.concatenate((arr, vals[:arr.shape[1]].reshape(1,arr.shape[1])))
    return arr

while True:
    logs = {}

    for dirname, dirnames, filenames in os.walk('../../exp/mpii'):
        for subdirname in dirnames:
            logs[subdirname] = {}
            train_path = '../../exp/mpii/' + subdirname + '/train.log'
            test_path = '../../exp/mpii/' + subdirname + '/test.log'
            if (os.path.exists(train_path) and os.path.exists(test_path) and
                os.stat(train_path).st_size != 0 and os.stat(test_path).st_size != 0):
                logs[subdirname]['train'] = readlog(train_path)
                logs[subdirname]['test'] = readlog(test_path)
                if track_multiple and exp_to_track in subdirname:
                    if not subdirname in experiments_to_show:
                        experiments_to_show += [subdirname]

    print "Updated experiments to show:",
    for exp in experiments_to_show: print exp,
    print ""

    idx = [1, 2, 0] # Epoch, Loss, Accuracy indices

    plt.clf() 
    
    fig = plt.figure(1, facecolor='w')
    last = 25
    last_str = '(last %d)' % last

    axs = {"Train loss":fig.add_subplot(421),
           "Test loss":fig.add_subplot(423),
           "Train accuracy":fig.add_subplot(425),
           "Test accuracy":fig.add_subplot(427),
           ("Train loss %s"%last_str):fig.add_subplot(422),
           ("Test loss %s"%last_str):fig.add_subplot(424),
           ("Train accuracy %s"%last_str):fig.add_subplot(426),
           ("Test accuracy %s"%last_str):fig.add_subplot(428)}

    for k in axs.keys():
        for tick in axs[k].xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in axs[k].yaxis.get_major_ticks():
            tick.label.set_fontsize(8)

        plt_idx = idx[0]
        if 'loss' in k: plt_idx = idx[1]
        if 'accuracy' in k: plt_idx = idx[2]

        start_idx = 0
        if last_str in k: start_idx = -last

        log_choice = 'train'
        if 'Test' in k: log_choice = 'test'

        max_x = -1
        for exp in experiments_to_show:
            log = logs[exp][log_choice]
            temp_start_idx = start_idx
            if abs(start_idx) > log.shape[0]: temp_start_idx = 0
            axs[k].plot(log[temp_start_idx:,idx[0]], log[temp_start_idx:,plt_idx], label=exp)
            if log[-1,idx[0]] > max_x:
                max_x = log[-1,idx[0]]

        if last_str in k:
            axs[k].set_xlim(max(0,max_x-last),max_x)
        else:
            axs[k].set_xlim(0,max_x)
        
        axs[k].set_title(k)
        if 'accuracy' in k and not last_str in k:
            axs[k].set_ylim(0,1)

    axs['Test accuracy'].legend(loc='lower right', fontsize=10)
    print time.strftime('%X %x %Z')
    plt.show()
    plt.pause(900)
