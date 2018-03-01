import sys, Image
from numpy import *
import scipy.ndimage

def score_from_autocorr(img0, img1, corres):
  # Code by Philippe Weinzaepfel
  # Compute autocorrelation
  # parameters
  sigma_image = 0.8 # for the gaussian filter applied to images before computing derivatives
  sigma_matrix = 3.0 # for the integration gaussian filter
  derivfilter = array([-0.5,0,0.5]) # function to compute the derivatives
  # smooth_images
  tmp = scipy.ndimage.filters.gaussian_filter1d(img0.astype(float32), sigma_image, axis=0, order=0, mode='nearest')
  img0_smooth = scipy.ndimage.filters.gaussian_filter1d(tmp, sigma_image, axis=1, order=0, mode='nearest')
  # compute the derivatives
  img0_dx = scipy.ndimage.filters.convolve1d(img0_smooth, derivfilter, axis=0, mode='nearest')
  img0_dy = scipy.ndimage.filters.convolve1d(img0_smooth, derivfilter, axis=1, mode='nearest')
  # compute the auto correlation matrix
  dx2 = sum(img0_dx*img0_dx,axis=2)
  dxy = sum(img0_dx*img0_dy,axis=2)
  dy2 = sum(img0_dy*img0_dy,axis=2)
  # integrate it
  tmp = scipy.ndimage.filters.gaussian_filter1d(dx2, sigma_matrix, axis=0, order=0, mode='nearest')
  dx2_smooth = scipy.ndimage.filters.gaussian_filter1d(tmp, sigma_matrix, axis=1, order=0, mode='nearest')
  tmp = scipy.ndimage.filters.gaussian_filter1d(dxy, sigma_matrix, axis=0, order=0, mode='nearest')
  dxy_smooth = scipy.ndimage.filters.gaussian_filter1d(tmp, sigma_matrix, axis=1, order=0, mode='nearest')
  tmp = scipy.ndimage.filters.gaussian_filter1d(dy2, sigma_matrix, axis=0, order=0, mode='nearest')
  dy2_smooth = scipy.ndimage.filters.gaussian_filter1d(tmp, sigma_matrix, axis=1, order=0, mode='nearest')  
  # compute minimal eigenvalues: it is done by computing (dx2+dy2)/2 - sqrt( ((dx2+dy2)/2)^2 + (dxy)^2 - dx^2*dy^2)
  tmp = 0.5*(dx2_smooth+dy2_smooth)
  small_eigen = tmp - sqrt( maximum(0,tmp*tmp + dxy_smooth*dxy_smooth - dx2_smooth*dy2_smooth)) # the numbers can be negative in practice due to rounding errors
  large_eigen = tmp + sqrt( maximum(0,tmp*tmp + dxy_smooth*dxy_smooth - dx2_smooth*dy2_smooth))
  # Compute weight as flow score: preparing variable
  #parameters
  sigma_image = 0.8 # gaussian applied to images
  derivfilter = array([1.0,-8.0,0.0,8.0,-1.0])/12.0 # filter to compute the derivatives
  sigma_score = 50.0 # gaussian to convert dist to score
  mul_coef = 10.0 # multiplicative coefficients
  # smooth images
  tmp = scipy.ndimage.filters.gaussian_filter1d(img0.astype(float32), sigma_image, axis=0, order=0, mode='nearest')
  img0_smooth = scipy.ndimage.filters.gaussian_filter1d(tmp, sigma_image, axis=1, order=0, mode='nearest')
  tmp = scipy.ndimage.filters.gaussian_filter1d(img1.astype(float32), sigma_image, axis=0, order=0, mode='nearest')
  img1_smooth = scipy.ndimage.filters.gaussian_filter1d(tmp, sigma_image, axis=1, order=0, mode='nearest')
  # compute derivatives
  img0_dx = scipy.ndimage.filters.convolve1d(img0_smooth, derivfilter, axis=0, mode='nearest')
  img0_dy = scipy.ndimage.filters.convolve1d(img0_smooth, derivfilter, axis=1, mode='nearest')
  img1_dx = scipy.ndimage.filters.convolve1d(img1_smooth, derivfilter, axis=0, mode='nearest')
  img1_dy = scipy.ndimage.filters.convolve1d(img1_smooth, derivfilter, axis=1, mode='nearest')
  # compute it
  res = []
  for pos0, pos1, score in corres:
    p0, p1 = tuple(pos0)[::-1], tuple(pos1)[::-1] # numpy coordinates
    dist = sum( abs(img0_smooth[p0]-img1_smooth[p1]) + abs(img0_dx[p0]-img1_dx[p1]) + abs(img0_dy[p0]-img1_dy[p1]) )
    score = mul_coef * sqrt( max(0,small_eigen[p0])) / (sigma_score*sqrt(2*pi))*exp(-0.5*square(dist/sigma_score));
    res.append((pos0,pos1,score))
  return res


if __name__=='__main__':
  args = sys.argv[1:]
  img0 = array(Image.open(args[0]).convert('RGB'))
  img1 = array(Image.open(args[1]).convert('RGB'))
  out = open(args[2]) if len(args)>=3 else sys.stdout
  
  ty0, tx0 = img0.shape[:2]
  ty1, tx1 = img1.shape[:2]
  rint = lambda s:  int(0.5+float(s))
  
  retained_matches = []
  for line in sys.stdin:
    line = line.split()
    if not line or len(line)!=6 or not line[0][0].isdigit():  continue
    x0, y0, x1, y1, score, index = line
    retained_matches.append(((min(tx0-1,rint(x0)),min(ty0-1,rint(y0))),
                             (min(tx1-1,rint(x1)),min(ty1-1,rint(y1))),0))
  
  assert retained_matches, 'error: no matches piped to this program'
  
  for p0, p1, score in score_from_autocorr(img0, img1, retained_matches):
    print >>out, '%d %d %d %d %f' %(p0[0],p0[1],p1[0],p1[1],score)




































