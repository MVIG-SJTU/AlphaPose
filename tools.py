import numpy as np
from scipy.stats import multivariate_normal as mm



def random_square(size):
  if isinstance(size, int):
    w = h = size
  else:
    w, h = size
  
  _m1 = (np.random.randint(w), np.random.randint(h))
  _r1 = np.random.rand(2, 2)  
  _s1 = _r1 @ _r1.T * np.random.randint(w//2, w)

  return mm(mean=_m1, cov=_s1)


def random_heatmap(size):
  if isinstance(size, int):
    w = h = size
  else:
    w, h = size
  
  k1 = random_square(size)
  k2 = random_square(size)

  xx, yy = np.meshgrid(np.array(range(w)), np.array(range(h)))
  pos = np.empty(xx.shape + (2, ))

  pos[:, :, 0] = xx; pos[:, :, 1] = yy
  _gama = np.random.rand()
  
  return _gama * k1.pdf(pos) + (1-_gama) * k2.pdf(pos)
