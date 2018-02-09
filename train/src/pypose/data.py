import numpy as np
import h5py
import getopt, sys, time

import img
import draw
import ref
import segment

def gendefault(annot, idx, img_in, chg=None):
    # Initialize sample parameters
    c = annot['center'][idx]
    s = annot['scale'][idx]
    flip, r = False, 0
    flip_idxs = ref.flipped_parts[annot.attrs['name']]

    # Handle data augmentation
    if chg is not None:
        # Flipping
        if 'flip' in chg:
            if np.random.rand() < .5:
                flip = True
        # Scaling
        if 'scale' in chg:
            s *= min(1+chg['scale'], max(1-chg['scale'], (np.random.randn() * chg['scale']) + 1))
        # Rotation
        if 'rotate' in chg:
            if chg['rotate'] == -1:
                # Force vertical orientation
                r = annot['torsoangle'][idx]
            else:
                r = np.random.randint(-chg['rotate'], chg['rotate'] + 1)
        # Translation
        if 'translate' in chg:
            for i in xrange(2):
                offset = np.random.randint(-chg['translate'], chg['translate'] + 1)
                c[i] += offset

    # Generate input image
    cropped = img.crop(img_in, c, s, ref.in_res, rot=r)
    inp = np.zeros((3, ref.in_res[0], ref.in_res[1]))
    for i in xrange(3):
        inp[i, :, :] = cropped[:, :, i]

    # Generate part heatmap output
    num_parts = annot['part'].shape[1]
    out = np.zeros((num_parts, ref.out_res[0], ref.out_res[1]))
    for i in xrange(num_parts):
        pt = annot['part'][idx,i]
        if pt[0] > 0:
            draw.gaussian(out[i], img.transform(pt, c, s, ref.out_res, rot=r), 2)

    # Flip sample
    if flip:
        inp = np.array([np.fliplr(inp[i]) for i in xrange(len(inp))])
        out = np.array([np.fliplr(out[flip_idxs[i]]) for i in xrange(len(out))])

    return inp,out

def gendetect(annot, idx, img_in, chg=None):
    img_c = [img_in.shape[1]/2, img_in.shape[0]/2]
    img_s = max(img_in.shape) / 200
    flip, r = False, 0
    idxs = np.where(annot['index'][:] == annot['index'][idx])[0]

    # Handle data augmentation
    if chg is not None:
        # Flipping
        if 'flip' in chg:
            if np.random.rand() < .5:
                flip = True
        # Scaling
        if 'scale' in chg:
            img_s *= min(1+chg['scale'], max(1-chg['scale'], (np.random.randn() * chg['scale']) + 1))
        # Rotation
        # if 'rotate' in chg:
        #     r = np.random.randint(-chg['rotate'], chg['rotate'] + 1)
        # Translation
        if 'translate' in chg:
            for i in xrange(2):
                offset = np.random.randint(-chg['translate'], chg['translate'] + 1)
                c[i] += offset

    img_c[0] += img_s * np.random.randint(-10,10)
    img_c[1] += img_s * np.random.randint(-10,10)
    cropped = img.crop(img_in, img_c, img_s, ref.in_res)
    inp = np.zeros((3, ref.in_res[0], ref.in_res[1]))
    for i in xrange(3): inp[i, :, :] = cropped[:, :, i]

    out = np.zeros((2, ref.out_res[0], ref.out_res[1]))
    for i in idxs:
        pt = img.transform(annot['center'][i], img_c, img_s, ref.out_res)
        draw.gaussian(out[0], pt, 1)
        out[1,pt[1]-1:pt[1]+1,pt[0]-1:pt[0]+1] = annot['scale'][i] / img_s

    if flip:
        inp = np.array([np.fliplr(inp[i]) for i in xrange(len(inp))])
        out = np.array([np.fliplr(out[i]) for i in xrange(len(out))])

    return inp,out

def gencascade(annot, idx, img_in, chg=None, extra_args=None):
    jnt1 = extra_args[0]
    jnt2 = extra_args[1]
    pt1 = annot['part'][idx,jnt1]
    pt2 = annot['part'][idx,jnt2]
    if pt1.min() <= 0 or pt2.min() <= 0:
        return np.zeros((3,ref.out_res[0],ref.out_res[1])), np.zeros((2,ref.out_res[0],ref.out_res[1]))
    else:
        return img.two_pt_crop(img_in, annot['scale'][idx], pt1, pt2, 1.8, ref.out_res, chg)

def gensample(annot, idx, chg=None, sampletype='default', extra_args=None):
    img_in = ref.loadimg(annot, idx)
    if sampletype == 'default':
        return gendefault(annot, idx, img_in, chg)
    elif sampletype == 'detect':
        return gendetect(annot, idx, img_in, chg)
    elif sampletype == 'cascade':
        return gencascade(annot, idx, img_in, chg, extra_args)

def generateset(dataset, settype, filename, numsamples, datadir=None, chg=None, sampletype='default', idxs=None, extra_args=None):
    # Generate full hdf5 dataset

    # Path to output file
    if datadir is None:
        filepath = ref.posedir + '/data/' + dataset + '/' + filename + '.h5'
    else:
        filepath = datadir + '/' + dataset + '/' + filename + '.h5'
    # Load in annotations
    annot = ref.load(dataset, settype)

    # Option to strictly follow the order of the provided annotations
    # Useful for generating test sets.
    if idxs is None:
        numavailable = len(annot['index']) # Number of available samples
    else:
        numavailable = len(idxs)
    inorder = False
    if numsamples == -1:
        numsamples = numavailable
        inorder = True

    print ""
    print "Generating %s %s set: %s" % (dataset, sampletype, settype)
    print "Path to dataset: %s" % filepath
    print "Number of samples: %d" % numsamples
    print "Data augmentation: %s" % (str(chg))

    # Data/label sizes can be all over the place, this is the easiest way to check
    ex_in, ex_out = gensample(annot, 0, chg=chg, sampletype=sampletype, extra_args=extra_args)

    # Initialize numpy arrays to hold data
    data = np.zeros((numsamples, ex_in.shape[0], ex_in.shape[1], ex_in.shape[2]), np.float32)
    label = np.zeros((numsamples, ex_out.shape[0], ex_out.shape[1], ex_out.shape[2]), np.float32)
    ref_idxs = np.zeros((numsamples, 1), np.float32)

    # Loop to generate new samples
    print ''
    print '| Progress            |'
    print '|',
    sys.stdout.flush()

    starttime = time.time()
    for i in xrange(numsamples):
        if idxs is not None: idx = idxs[i]
        elif inorder: idx = i
        else: idx = np.random.randint(numavailable)

        data[i], label[i] = gensample(annot, idx, chg=chg, sampletype=sampletype, extra_args=extra_args)
        ref_idxs[i] = idx

        if i % (numsamples/10) == 0:
            print '=',
            sys.stdout.flush()

    print '|'
    print ''
    print 'Done!',
    print '(%.2f seconds to complete.)' % (time.time() - starttime)
    print ''

    # Write out to hdf5 files
    with h5py.File(filepath, 'w') as f:
        f['data'] = data
        f['label'] = label
        f['index'] = ref_idxs

def helpmessage():
    print "Extra flags:"
    print "  -d, --dataset    :: Datset choice (mpii or flic), REQUIRED"
    print "  -o, --outfile    :: Output file for data (do not include '.h5'), REQUIRED"
    print "  -p, --prefix     :: Directory to save to (no need to include dataset name)"
    print "  -t, --type       :: Dataset type (train or test), default is train"
    print "  -n, --numsamples :: Number of samples to generate, default is all available (-1) for test and 100 for train"
    print ""
    print "Augmentation options: (default Tompson's options for train, none for test)"
    print "  -m, --move       :: Translate (0 - 50)"
    print "  -z, --zoom       :: Scale (0.0 - 1.0)"
    print "  -r, --rotate     :: Rotate (-1 for fixed vertical, 0-180 for max distortion)"
    print "  (Tompson's options are: -m 0 -z .5 -r 20"
    print ""
    print "Other dataset types:"
    print "  -q, --detect"
    print "  -c, --cascade     :: Provide first joint as argument, must use additional argument below"
    print "  -j, --pairedjoint :: Provide second joint to be used with 'cascade'"
    print ""
    print "Additional limb heatmap output:"
    print "  -s, --segment    :: - 0 No limb segment output (default)"
    print "                      - 1 Does not distinguish parts, angle == angle + 180"
    print "                      - 2 Distinguishes different part types, angle == angle + 180"
    print "                      - 3 Distinguishes different part types, angle != angle + 180"
    sys.exit(2)

def main(argv):
    # Default values
    dataset = None
    datadir = None
    outfile = None
    numsamples = 100
    settype = 'train'
    chg = None
    sampletype = 'default'
    jnt1 = -1
    jnt2 = -1
    extra = None

    # Process command line arguments
    try:
        opts, args = getopt.getopt(argv, "hd:o:p:t:n:m:z:r:s:qc:j:", ["help", "dataset=", "outfile=", "prefix=", "type=",
                                                                   "numsamples=", "move=", "zoom=", "rotate=",
                                                                   "segment=", "detect", "cascade=", "pairedjoint="])
    except getopt.GetoptError:
        print "Incorrect arguments"
        helpmessage()
        sys.exit()
    for opt,arg in opts:
        # Help
        if opt in ('-h','--help'):
            helpmessage()
        # Dataset choice
        elif opt in ('-d','--dataset'):
            dataset = arg
            if not (dataset in ['mpii', 'flic']):
                print "Bad argument for --dataset"
                helpmessage()
        # Output file
        elif opt in ('-o','--outfile'):
            outfile = arg
        # Prefix
        elif opt in ('-p','--prefix'):
            datadir = arg
        # Set type
        elif opt in ('-t','--type'):
            settype = arg
            if not (settype in ['train','test','valid','train_obs','test_obs']):
                print "Bad argument for --type"
                helpmessage()
        # Number of samples
        elif opt in ('-n','--numsamples'):
            numsamples = int(arg)
            if numsamples < -1:
                print "Bad argument for --numsamples"
                helpmessage()
        # Move
        elif opt in ('-m','--move'):
            move = int(arg)
            if not 0 <= move <= 50:
                print "Bad argument for --move"
                helpmessage()
            else:
                if chg is None:
                    chg = {}
                chg['translate'] = move
        # Zoom
        elif opt in ('-z','--zoom'):
            zoom = float(arg)
            if not 0 <= zoom <= 1:
                print "Bad argument for --zoom"
                helpmessage()
            else:
                if chg is None:
                    chg = {}
                chg['scale'] = zoom
        # Rotate
        elif opt in ('-r','--rotate'):
            rot = int(arg)
            if not -1 <= rot <= 180:
                print "Bad argument for --rotate"
                helpmessage()
            else:
                if chg is None:
                    chg = {}
                chg['rotate'] = rot
        # Segment
        elif opt in ('-s','--segment'):
            seg = int(arg)
            if not (0 <= seg <= 3):
                print "Bad argument for --segment"
                helpmessage()
        # Detect
        elif opt in ('-q','--detect'):
            sampletype = 'detect'
        # Cascade
        elif opt in ('-c','--cascade'):
            sampletype = 'cascade'
            jnt1 = int(arg)
        elif opt in ('-j','--pairedjoint'):
            jnt2 = int(arg)

    if dataset is None:
        print "No dataset chosen."
        helpmessage()
    if outfile is None:
        print "No output filename chosen."
        helpmessage()

    if settype in ['test','test_obs']:
        # Test set has a standard number of images, and no augmentation
        numsamples = -1
    elif settype == 'train' and chg is None:
        if sampletype == 'default': chg = {'rotate':20, 'scale':.5}
        elif sampletype == 'cascade': chg = {'rotate':20,'scale':.2, 'translate':20}
        else: chg = {}
        chg['flip'] = True

    # If we're generating cascade data make sure two joints have been provided
    if sampletype == 'cascade':
        if jnt1 == -1 or jnt2 == -1:
            print "Need two joints to generate cascade data"
            helpmessage()
        extra = [jnt1, jnt2]

    generateset(dataset, settype, outfile, numsamples, datadir=datadir, chg=chg, sampletype=sampletype, extra_args=extra)

if __name__ == "__main__":
    main(sys.argv[1:])
