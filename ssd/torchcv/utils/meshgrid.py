import torch


def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)
