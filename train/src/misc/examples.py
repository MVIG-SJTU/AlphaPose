# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# All of these examples are really, really outdated but offer some insights 
# into using the python code, if you want to check it out
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import numpy as np
import pypose as pose
import pypose.mpii as ds  # Use this to swap which dataset you want to use

# Sample dataset generation
if False:
    # Everything is pretty self explanatory here (outdated and not functional anymore
    # command line interface is better)
    filename = 'sample'
    numsamples = 100
    is_train = 1
    augmentation = 1
    pose.data.generateset(ds, filename, numsamples, is_train, chg=augmentation)

# Sample report
# (compares performance based on whether the person is facing forward or backward)
if False:
    # Get predictions
    preds = np.load(pose.eval.get_path(ds.name, 'nyu_pred'))
    # Get prediction error
    dists = pose.eval.getdists(preds)

    # To create our filters for the report: 307-DR,304-hg-D
    # Load up ground truth annotations
    gt_idx = pose.eval.gt_idx[ds.name]
    # Compare shoulder annotations
    sho_diff = np.array([ds.partinfo(gt_idx[i,0],gt_idx[i,1],'lsho')[0][0] -
                         ds.partinfo(gt_idx[i,0],gt_idx[i,1],'rsho')[0][0]
                         for i in xrange(len(gt_idx))], np.float)
    # Normalize difference by sample scale size
    sho_diff /= gt_idx[:,2]
    # Define the filters, numpy generates boolean arrays out of these comparisons
    filtnames = ['Forward', 'Back', 'Profile', 'Total']
    thresh = .3
    filts = [sho_diff > thresh,
             sho_diff < -thresh,
             (sho_diff < thresh) * (sho_diff > -thresh),
             None]

    # Prepare the document
    title='Performance Comparison - Facing Forward or Backward'
    pdf = pose.report.PdfPages(pose.ref.posedir+'/img/reports/fwd_back_sample.pdf')

    # Add whatever pages you want
    print "Doing overall comparison..."
    pose.report.filtercomparison(ds.name, dists, filts, filtnames=filtnames, title=title, pdf=pdf)
    for i,filt in enumerate(filts[:-1]):
        print "Generating images for - %s..." % filtnames[i]
        pose.report.sampleimages(ds, preds, dists=dists, pdf=pdf, title=filtnames[i], filt=filt)
        pose.report.sampleimages(ds, preds, dists=dists, pdf=pdf, title=filtnames[i], filt=filt, get_worst=True)

    # Save the pdf
    pdf.close()

if True:
    # Get predictions
    preds = np.load(pose.eval.get_path(ds.name, 'nyu_pred'))
    # Get prediction error
    dists = pose.eval.getdists(preds)

    # To create our filters for the report:
    # Load up ground truth annotations
    gt_idx = pose.eval.gt_idx[ds.name]
    # Calculate torso angles (note this only works for mpii)
    torso_angles = np.array([abs(ds.torsoangle(gt_idx[i,0], gt_idx[i,1])) for i in xrange(len(gt_idx))])
    # Define filters
    filtnames = ['< 20 degrees','20 < 40','40 < 120', '> 120', 'Total']
    filts = [torso_angles <= 20,
             (20 < torso_angles) * (torso_angles < 40),
             (40 < torso_angles) * (torso_angles < 120),
             (120 < torso_angles),
             None]

    # Prepare the document
    title='Performance Comparison - Torso Deviation from Vertical'
    pdf = pose.report.PdfPages(pose.ref.posedir+'/img/reports/torso_angle_sample.pdf')

    print "Doing overall comparison..."
    pose.report.filtercomparison(ds.name, dists, filts, filtnames=filtnames, title=title, pdf=pdf)
    for i in xrange(7):
        # This loop will only generate poor performing images for the first filter (people who are upright)
        print "Generating images for page - %d..." % i
        pose.report.sampleimages(ds, preds, dists=dists, pdf=pdf, title=filtnames[0], filt=filts[0],
                                get_worst=True, page_num=i+1)

    # Save the pdf
    pdf.close()


"""
overall performance - taken out of report.py not adjusted to work here

def make(dataset, preds, partnames=None):
    pdf = PdfPages(ref.posedir+'/img/test.pdf')

    num_pages = 10
    dists = eval.getdists(preds)

    for i in xrange(num_pages):
        print "Page %d..." % i
        page_choice = i + 1
        if i < num_pages / 2:
            get_worst = False
        else:
            page_choice -= num_pages / 2
            get_worst = True
        sampleimages(dataset, preds, dists=dists, pdf=pdf, get_worst=get_worst,
                     partnames=partnames, title='Overall Performance', page_num=page_choice)

    pdf.close()

"""
