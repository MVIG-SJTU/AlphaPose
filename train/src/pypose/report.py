import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys, getopt
import h5py

import img
import eval
import ref
import data

# =============================================================================
# Helper functions
# =============================================================================

# loss_idxs = [5,4]
loss_idxs = [2,2]
acc_idxs = [3,5]

def doit(train_log,test_log,idx=None):
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    plottraintest(ax, train_log, test_log, idx=idx, title='Accuracy')
    plt.show()
    plt.savefig('whatever.png')
    plt.clf()
    return

def plottraintest(ax, train_log, test_log, idx=loss_idxs, title='Loss'):
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

def tabletext(ax, txt, r, c, align='center', size=10):
    # Shift left/right for alignment
    if align == 'left': c -= .4
    elif align == 'right': c += .4
    # Handle weirdness because the first column is double width
    if c > 0: c += 1
    elif align == 'center': c += .5
    elif align == 'right': c += 1
    # Write text
    ax.text(c+.5, .25-r, txt, fontsize=size, horizontalalignment=align)

def setuptable(ax, numrows, numcols, row_labels=None, col_labels=None):
    # Initial table set up
    # (first column is double width to allow more room for label names)
    for r in xrange(numrows):
        if row_labels is not None and r < len(row_labels):
            tabletext(ax, row_labels[r], r+1, 0, align='left')
        if r: thk = 1
        else: thk = 2
        plt.plot([0,numcols+1],[-r,-r],color='k',linewidth=thk)
    for c in xrange(numcols):
        if col_labels is not None and c < len(col_labels):
            tabletext(ax, col_labels[c], 0, c)
        if c: thk = 1
        else: thk = 2
        plt.plot([c+2, c+2],[1,1-numrows],color='k',linewidth=thk)
    ax.set_xlim(0,numcols+1)
    ax.set_ylim(-numrows+1,1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def setuppdjplot(ax, i, num_parts, num_cols):
    # Configuration of ticks in plots
    major_ticks_y = np.arange(0,1.01,.2)
    minor_ticks_y = np.arange(0,1.01,.1)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        if i < num_parts - num_cols:
            tick.label.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        if not (i % num_cols == 0):
            tick.label.set_visible(False)
    ax.grid()
    ax.grid(which='minor', alpha=0.5)

def loadpreds(dataset, predfile, pred_opts, get_hms=False, dotrain=False):
    num_parts, vert, obs = pred_opts
    hms = None
    with h5py.File(predfile, 'r+') as f:
        # Choose appropriate key
        if vert: k = 'preds_v'
        else: k = 'preds_tf'
        # Load predictions
        if k in f.keys():
            preds = np.array(f[k])
        else:
            preds = eval.transformpreds(dataset, f['preds'],
                                        [64, 64], rot=vert)
            f[k] = preds

        # Ignore additional predictions from segmentation (soon to be unnecessary)
        if preds.shape[1] > num_parts:
            preds = preds[:,:num_parts,:]
        # Also load heatmaps if necessary
        if get_hms:
            hms = np.array(f['preds_raw'])

        # Load distances
        dist_key = 'dist_'
        if vert: dist_key += 'v'
        if obs: dist_key += 'o'
        if dist_key in f.keys():
            dists = np.array(f[dist_key])
        else:
            # Evaluate distances to ground truth
            dists = eval.getdists(preds, dotrain)
            f[dist_key] = dists
    return preds, dists, hms

# =============================================================================
# Page creation functions
# =============================================================================

# Sample images page
#   - m x n rows and columns of skeleton images
#   - allow filters and sort by overall score or score for specific parts
#   - input res, number of images
default_res = [256, 256]
def sampleimages(annot, preds, dists=None, partnames=None, filt=None, num_rows=7, num_cols=5, res=default_res, get_worst=False, page_num=1, pdf=None, title='Prediction Examples'):
    # Dataset name
    ds = annot.attrs['name']

    # Initialize blank page
    plt.clf()
    fig = plt.figure(figsize=(8.5,11), dpi=100, facecolor='w')
    ax = fig.add_subplot(111)
    page = np.zeros((res[0]*num_rows, res[1]*num_cols, 3), np.uint8)

    # If no specific parts have been chosen, use them all for scoring
    if partnames is None:
        partnames = ref.parts[ds]
    part_idxs = [ref.parts[ds].index(part) for part in partnames]
    part_filt = [i in part_idxs for i in xrange(len(ref.parts[ds]))]

    # If no filter is provided create entirely true array
    if filt is None:
        filt = np.array([True for _ in xrange(len(preds))])
    else:
        filt = filt.copy()

    # If no precalculated distances are provided, calculate them
    if dists is None:
        dists = eval.getdists(preds)

    # Determine scores from which we'll sort the images
    scores = np.zeros(len(preds))
    for i in xrange(len(preds)):
        # A bit of an interesting line below, gets the mean distance for a particular image
        # while only considering the parts we want and ignoring any parts where there's no annotation
        vals = dists[i, part_filt * (annot['part'][i,:,0] > 0)]
        if len(vals) > 0:
            scores[i] = vals.mean()
        else:
            # If no valid annotation to make a score, filter out this example
            filt[i] = False
    if get_worst:
        # Flip the scores if we're getting the worst images
        scores = -scores
    best_idxs = scores.argsort()
    curr_idx = 0

    # Start filling in the overall image
    for i in xrange(page_num * num_rows * num_cols):
        while curr_idx < len(best_idxs) and not filt[best_idxs[curr_idx]]:
            curr_idx += 1
        if curr_idx >= len(best_idxs): break

        # If we're doing multiple pages, pass over the images that have already been used
        if i >= (page_num - 1) * num_rows * num_cols:
            idx = best_idxs[curr_idx]
            curr_pred = preds[idx].copy()
            curr_pred[-(part_filt * (annot['part'][idx,:,0] > 0)), :] = -1000
            new_img = img.sample_with_skeleton(annot, idx, curr_pred, res=res)
            row = ((i % (num_rows * num_cols)) / num_cols) * res[0]
            col = ((i % (num_rows * num_cols)) % num_cols) * res[1]

            page[row:row+res[0], col:col+res[1]] = new_img
        curr_idx += 1

    # Plot management
    if not get_worst:
        title += ' - Best - '
    else:
        title += ' - Worst - '
    title += 'Page %d' % page_num
    ax.set_title(title)
    ax.imshow(page)
    ax.axis('off')
    fig.subplots_adjust(left=0.05,right=.95,bottom=0.05,top=.95)
    if pdf:
        pdf.savefig()
    else:
        plt.show()

# Part heatmaps page
def partheatmaps(annot, preds, preds_raw, dists=None, partnames=None, filt=None, num_rows=7, vert=False, num_cols=2, res=default_res, get_worst=False, page_num=1, pdf=None, title='Prediction Examples'):
    # Dataset name
    ds = annot.attrs['name']

    # Initialize blank page
    plt.clf()
    fig = plt.figure(figsize=(8.5,11), dpi=100, facecolor='w')
    ax = fig.add_subplot(111)

    # If no specific parts have been chosen, use them all for scoring
    if partnames is None:
        partnames = ref.parts[ds]
    part_idxs = [ref.parts[ds].index(part) if type(part) is str else part for part in partnames]
    part_filt = [i in part_idxs for i in xrange(len(ref.parts[ds]))]
    page = np.zeros((res[0]*num_rows, res[1]*num_cols*(1+len(part_idxs)), 3), np.uint8)

    # If no filter is provided create entirely true array
    if filt is None:
        filt = np.array([True for _ in xrange(len(preds))])
    else:
        filt = filt.copy()

    # If no precalculated distances are provided, calculate them
    if dists is None:
        dists = eval.getdists(preds)

    # Determine scores from which we'll sort the images
    scores = np.zeros(len(preds))
    for i in xrange(len(preds)):
        # A bit of an interesting line below, gets the mean distance for a particular image
        # while only considering the parts we want and ignoring any parts where there's no annotation
        vals = dists[i, part_filt * (annot['part'][i,:,0] > 0)]
        if len(vals) > 0:
            scores[i] = vals.mean()
        else:
            # If no valid annotation to make a score, filter out this example
            filt[i] = False
    if get_worst:
        # Flip the scores if we're getting the worst images
        scores = -scores
    best_idxs = scores.argsort()
    if title[:4] == 'head' and get_worst:
        np.save('worst_head_idxs',best_idxs[:200])
    curr_idx = 0

    # Start filling in the overall image
    for i in xrange(page_num * num_rows * num_cols):
        while curr_idx < len(best_idxs) and not filt[best_idxs[curr_idx]]:
            curr_idx += 1
        if curr_idx >= len(best_idxs): break

        # If we're doing multiple pages, pass over the images that have already been used
        if i >= (page_num - 1) * num_rows * num_cols:
            idx = best_idxs[curr_idx]
            if vert:
                inp, _ = data.gensample(annot, idx, chg={'rotate':-1})
            else:
                inp, _ = data.gensample(annot, idx)
            new_img = img.sample_with_heatmap(ds, inp, preds_raw[idx], num_rows=1, parts_to_show=part_idxs)
            row = ((i % (num_rows * num_cols)) / num_cols) * res[0]
            col = ((i % (num_rows * num_cols)) % num_cols) * res[1] * (1+len(part_idxs))

            page[row:row+res[0], col:col+(res[1]*(1+len(part_idxs)))] = new_img
        curr_idx += 1

    # Plot management
    if not get_worst:
        title += ' - Best - '
    else:
        title += ' - Worst - '
    title += 'Page %d' % page_num
    ax.set_title(title)
    ax.imshow(page)
    ax.axis('off')
    fig.subplots_adjust(left=0.05,right=.95,bottom=0.05,top=.95)
    if pdf:
        pdf.savefig()
    else:
        plt.show()
    fig.clf()

# Filter comparison page
#   - inputs a list of filters
#   - top row is overall performance comparison, and table with numbers including # samples
#   - performance broken down by part categories
def filtercomparison(dataset, dists, filts, filtnames, pdf=None, title='Performance comparison', other_dists=None, parts_to_show=None):
    # Initialize blank page
    # plt.clf()
    fig = plt.figure(figsize=(8.5,11), dpi=100, facecolor='w')

    if parts_to_show is None:
        part_labels = ref.pair_names[dataset]
        parts_to_show = ref.part_pairs[dataset]
    else:
        part_labels = [parts_to_show[i][0] for i in xrange(len(parts_to_show))]

    # Configuration of ticks in plots
    major_ticks_y = np.arange(0,1.01,.2)
    minor_ticks_y = np.arange(0,1.01,.1)

    #-------------------------------------------------------------------
    # Table with performance numbers
    #-------------------------------------------------------------------

    ax_table = fig.add_subplot(5,1,1)

    cols = ['', '#', 'Full'] + part_labels
    rows = ['Label'] + filtnames
    num_samples = [len(dists) if filt is None else filt.sum() for filt in filts]
    num_samples = [''] + num_samples
    if other_dists is not None:
        rows += other_dists.keys()
        num_samples += [len(dists) if other_dists[k][1] is None else other_dists[k][1].sum()
                        for k in other_dists.keys()]

    # Initial table set up
    for r in xrange(len(rows)):
        # Filter labels
        ax_table.text(.1,len(rows)-.75-r,rows[r],fontsize=10,horizontalalignment='left')
        # Number of samples available from each filter
        ax_table.text(2.5,len(rows)-.75-r,num_samples[r],fontsize=10,horizontalalignment='center')
        if r < len(rows) - 1:
            thk = 1
        else:
            thk = 2
        plt.plot([0,len(cols)+1],[r,r],color='k',linewidth=thk)

    for c in xrange(1, len(cols) + 1):
        ax_table.text(c+.5,len(rows)-.75,cols[c-1],fontsize=10,horizontalalignment='center')
        if c > 2:
            thk = 1
        else:
            thk = 2
        if c < len(cols):
            plt.plot([c+1, c+1],[0,len(rows)],color='k',linewidth=thk)

    # Performance numbers get filled in as we create the PDJ charts

    ax_table.set_xlim(0,len(cols)+1)
    ax_table.get_xaxis().set_visible(False)
    ax_table.get_yaxis().set_visible(False)

    ax_table.set_title(title, y=1.05)

    #-------------------------------------------------------------------
    # Overall performance chart
    #-------------------------------------------------------------------
    ax = fig.add_subplot(5,3,4)

    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        tick.label.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    ax.grid()
    ax.grid(which='minor', alpha=0.5)

    for i,filt in enumerate(filts):
        d, t = eval.pdjdata(dataset, dists, filt=filt)
        # Plot PDJ curve
        ax.plot(t,d)
        # Display performance number in table
        ax_table.text(3.5, len(rows)-i-1.75, '%04.1f' % (d[-1]*100),
                      fontsize=10, horizontalalignment='center')
    if other_dists is not None:
        for i,k in enumerate(other_dists):
            d, t = eval.pdjdata(dataset, other_dists[k][0], filt=other_dists[k][1])
            # Plot PDJ curve
            ax.plot(t,d)
            # Display performance number in table
            ax_table.text(3.5, len(rows)-i-len(filts)-1.75, '%04.1f' % (d[-1]*100),
                          fontsize=10, horizontalalignment='center')

    ax.set_ylim([0,1])

    #    box = ax.get_position()
    #    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax_labels = filtnames
    if other_dists is not None:
        ax_labels += other_dists.keys()
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), labels=ax_labels, fontsize=11)
    ax.set_title('Overall', fontsize=10)

    #-------------------------------------------------------------------
    # Separate charts for each part
    #-------------------------------------------------------------------
    num_cols = 3
    for i,pts in enumerate(parts_to_show):
        ax = fig.add_subplot(5,num_cols,i+7)
        setuppdjplot(ax, i, len(parts_to_show), num_cols)

        for j,filt in enumerate(filts):
            d, t = eval.pdjdata(dataset, dists, filt=filt, partnames=pts)
            # Plot PDJ curve
            ax.plot(t,d)
            # Display performance number in table
            ax_table.text(i+4.5, len(rows)-j-1.75, '%04.1f' % (d[-1]*100), fontsize=10, horizontalalignment='center')
        if other_dists is not None:
            for j,k in enumerate(other_dists):
                d, t = eval.pdjdata(dataset, other_dists[k][0], filt=other_dists[k][1], partnames=pts)
                # Plot PDJ curve
                ax.plot(t,d)
                # Display performance number in table
                ax_table.text(i+4.5, len(rows)-j-1.75-len(filts), '%04.1f' % (d[-1]*100),
                              fontsize=10, horizontalalignment='center')

        ax.set_title(part_labels[i], fontsize=10)
        ax.set_ylim([0,1])

    if pdf:
        pdf.savefig()
    else:
        plt.show()

# Training report page
#   - plot showing loss train/test across iterations
#   - accuracy vs state of the art for shoulders, elbows, wrists
#   - in plots also include (previous best benchmark)
#   - quite a lot of overlap with filter comparison code, a bit sloppy
#     there's probably a better code interface to design to pull this altogether
def trainingoverview(dataset, dists, filts, filtnames, pdf=None,
                     other_dists=None, parts_to_show=None, exp_id='default'):
    # Initialize blank page
    fig = plt.figure(figsize=(8.5,11), dpi=100, facecolor='w')

    # Default parts to show performance results
    if parts_to_show is None:
        parts_to_show = []
        if dataset == 'flic': parts_to_show += [('Face',['leye','reye','nose'])]
        else: parts_to_show += [('Head',['head','neck']),
                                ('Ank',['lank','rank']),
                                ('Knee',['lkne','rkne'])]
        parts_to_show += [('Sho',['lsho','rsho']),
                          ('Elb',['lelb','relb']),
                          ('Wri',['lwri','rwri']),
                          ('Hip',['lhip','rhip'])]
    part_labels = [parts_to_show[i][0] for i in xrange(len(parts_to_show))]

    # Load training logs
    log_dir = ref.posedir + '/exp/' + dataset + '/' + exp_id
    train_log = np.loadtxt(log_dir + '/train.log', skiprows=1)
    test_log = np.loadtxt(log_dir + '/test.log', skiprows=1)
    # Plot loss
    ax = fig.add_subplot(4,1,2)
    plottraintest(ax, train_log, test_log)

    # Setup table to hold performance numbers
    ax_table = fig.add_subplot(4,1,1)
    cols = ['Label'] + part_labels
    rows = filtnames
    if other_dists is not None:
        rows += other_dists.keys()
    setuptable(ax_table, len(rows)+1, len(cols), row_labels=rows, col_labels=cols)
    ax_table.set_title('%s - Experiment: %s' % (dataset.upper(), exp_id), y=1.05)

    # Generate PDJ charts for each part
    num_cols = 4
    for i,pts in enumerate(parts_to_show):
        ax = fig.add_subplot(4,num_cols,i+2*num_cols+1)
        setuppdjplot(ax, i, len(parts_to_show), num_cols)

        for j,filt in enumerate(filts):
            d, t = eval.pdjdata(dataset, dists, filt=filt, partnames=pts[1])
            # Plot PDJ curve
            ax.plot(t,d,label=filtnames[j])
            # Display performance number in table
            tabletext(ax_table, '%04.1f'%(d[-1]*100), j+1, i+1)
        if other_dists is not None:
            for j,k in enumerate(other_dists):
                d, t = eval.pdjdata(dataset, other_dists[k][0], filt=other_dists[k][1], partnames=pts[1])
                # Plot PDJ curve
                ax.plot(t,d,label=k)
                # Display performance number in table
                tabletext(ax_table, '%04.1f'%(d[-1]*100), j+1+len(filts), i+1)

        ax.set_title(pts[0], fontsize=10)
        ax.set_ylim([0,1])

        if i == len(parts_to_show) - 1:
            ax_labels = filtnames
            if other_dists is not None:
                ax_labels += other_dists.keys()
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=11)

    if pdf: pdf.savefig()
    else: plt.show()

# Limb report page
# A super basic report that just includes training/test loss and accuracy over time
def limbreport(dataset, exp_id, pdf=None):
    fig = plt.figure(figsize=(8.5,11), dpi=100, facecolor='w')

    # Plot train/test loss
    ax = fig.add_subplot(3,1,1)
    exp_dir = ref.posedir + '/exp/' + dataset + '/' + exp_id
    train_log = np.loadtxt(exp_dir + '/train.log', skiprows=1)
    test_log = np.loadtxt(exp_dir + '/test.log', skiprows=1)
    plottraintest(ax, train_log, test_log)

    # Plot test accuracy over time
    ax = fig.add_subplot(3,1,2)
    plottraintest(ax, train_log, test_log, idx=acc_idxs, title='Accuracy')

    # Plot precision/recall curve
    with h5py.File(exp_dir + '/preds.h5', 'r') as f:
        preds = np.array(f['preds_raw'])
    with h5py.File(ref.posedir + '/data/mpii/sho_neck_test.h5','r') as f:
        label = np.array(f['label'])

    p,r = [],[]
    for thrsh in np.arange(0,1.01,.05):
        preds_bool = np.floor(preds+thrsh)
        tp = (preds_bool * label).sum()
        fp = ((preds_bool - label) == 1).sum()
        fn = ((label - preds_bool) == 1).sum()
        p += [float(tp)/(tp+fp) if tp > 0 else 1]
        r += [float(tp)/(tp+fn)]
    ax = fig.add_subplot(3,1,3)
    ax.plot(r,p)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    if pdf:
        pdf.savefig()
    else:
        plt.show()

# =============================================================================
# Main command line interface
# =============================================================================

def helpmessage():
    print "This isn't too helpful, updating message soon..."
    sys.exit(2)

# Main
def main(argv):
    dataset = None
    exp_id = None
    extra = []
    prev = []
    other_dists = {}
    vert = False
    images = False
    obs = False
    limb = False

    # Process command line arguments
    try:
        opts, args = getopt.getopt(argv, "hd:e:c:p:viol", ["help", "dataset=", "expID=", "compare=", "prev=",
                                                           "vert", "images", "obs", "limb"])
    except getopt.GetoptError:
        print "Incorrect arguments"
        helpmessage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            helpmessage()
        elif opt in ('-d', '--dataset'): dataset = arg
        elif opt in ('-e', '--expID'): exp_id = arg
        elif opt in ('-c', '--compare'): extra += arg.split(',')
        elif opt in ('-p', '--prev'): prev += arg.split(',')
        elif opt in ('-v', '--vert'): vert = True
        elif opt in ('-i', '--images'): images = True
        elif opt in ('-o', '--obs'): obs = True
        elif opt in ('-l', '--limb'): limb = True

    if dataset is None:
        print "No dataset chosen."
        helpmessage()
    if not (dataset in ['mpii','flic']):
        print "Bad argument for --dataset"
        helpmessage()
    if exp_id is None:
        print "No experiment number provided."
        helpmessage()
    expdir = ref.posedir + '/exp/' + dataset + '/' + exp_id

    # Generate the simple report for mini limb networks
    if limb:
        pdf = PdfPages(expdir + '/report.pdf')
        limbreport(dataset, exp_id, pdf)
        pdf.close()
        return

    # Load in dataset information
    num_parts = len(ref.parts[dataset])
    if obs:
        annot = ref.load(dataset, 'test_obs')
        eval.annot[dataset] = annot
    else:
        annot = ref.load(dataset, 'valid')

    # Load predictions
    print "Loading predictions"
    pred_opts = [num_parts, vert, obs]
    preds, dists, _ = loadpreds(dataset, expdir + '/preds.h5', pred_opts, images)

    # Load previous predictions
    for prv in prev:
        _,d,_ = loadpreds(dataset, expdir + '/preds_%s.h5' % prv, pred_opts)
        other_dists[prv] = [d, None]

    # Load comparison predictions
    for ext in extra:
        predfile = ref.posedir + '/exp/' + dataset + '/' + ext + '/preds.h5'
        _,d,_ = loadpreds(dataset, predfile, pred_opts)
        other_dists[ext] = [d, None]

    # Load previous best
    if vert: predfile = expdir + '/../best/preds_vert.h5'
    else: predfile = expdir + '/../best/preds.h5'
    _,best_dists,_ = loadpreds(dataset, predfile, pred_opts)
    #other_dists["Kaiyu's best model"] = [best_dists, None]

    # Load NYU predictions
    if dataset == 'mpii':
        nyu_dists = np.load(eval.get_path(dataset, 'nyu_dists'))
    else:
        if not obs: nyu_preds = np.load(eval.get_path(dataset, 'nyu_pred'))
        else: nyu_preds = np.load(eval.get_path(dataset, 'nyu_pred_obs'))
        nyu_dists = eval.getdists(nyu_preds)
        np.save('nyu_dists_%s%s'%(dataset,'_obs' if obs else ''),nyu_dists)
    other_dists['Tompson'] = [nyu_dists, None]

    # Load training set predictions
    if False:
        _,d,_ = loadpreds(dataset, expdir + '/preds_train.h5', pred_opts, dotrain=True)
        other_dists['Train'] = [d, None]

    filt = None

    print "Creating overview page"
    # Main report creation
    pdf = PdfPages(expdir + '/report.pdf')

    # Training overview page
    trainingoverview(dataset, dists, [filt], [exp_id], exp_id=exp_id,
                     other_dists=other_dists, pdf=pdf)

    if images:
        print "Creating prediction examples page"
        # Overall performance examples
        num_good_exs = 2
        num_bad_exs = 6
        for i in xrange(num_good_exs):
            sampleimages(annot,preds,dists,pdf=pdf,page_num=i+1)
        for i in xrange(num_bad_exs):
            sampleimages(annot,preds,dists,get_worst=True,pdf=pdf,page_num=i+1)

        # print "Creating part heatmap examples"
        # # Heatmap examples
        # for i in xrange(len(ref.part_pairs[dataset])):
        #     title = ref.pair_names[dataset][i]
        #     pt_names = ref.part_pairs[dataset][i]
        #     if not title == 'face':
        #         partheatmaps(annot,preds,preds_raw,dists=dists,partnames=pt_names,title='%s Heatmap Examples'%title,
        #                      pdf=pdf, page_num=1, vert=vert)
        #         for j in xrange(1,3):
        #             partheatmaps(annot,preds,preds_raw,dists=dists,partnames=pt_names,title='%s Heatmap Examples'%title,
        #                          pdf=pdf, page_num=j, vert=vert, get_worst=True)

    pdf.close()

if __name__ == "__main__":
    main(sys.argv[1:])

# Reference for creating a left/right filter
# filt = np.array([ref.partinfo(annot,i,'lsho')[0] >
#                  ref.partinfo(annot,i,'rsho')[0]
#                  for i in xrange(len(annot['index']))])
