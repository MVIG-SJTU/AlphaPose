import sys
from PIL import Image
from numpy import *
from matplotlib.pyplot import *


def show_correspondences( img0, img1, corr ):
    assert corr.shape[-1]==6
    corr = corr[corr[:,4]>0,:]
    
    # make beautiful colors
    center = corr[:,[1,0]].mean(axis=0) # array(img0.shape[:2])/2 #
    corr[:,5] = arctan2(*(corr[:,[1,0]] - center).T)
    corr[:,5] = int32(64*corr[:,5]/pi) % 128
    
    set_max = set(corr[:,5])
    colors = {m:i for i,m in enumerate(set_max)}
    colors = {m:cm.hsv(i/float(len(colors))) for m,i in colors.items()}
    
    def motion_notify_callback(event):
      if event.inaxes==None: return
      numaxis = event.inaxes.numaxis
      if numaxis<0: return
      x,y = event.xdata, event.ydata
      ax1.lines = []
      ax2.lines = []
      n = sum((corr[:,2*numaxis:2*(numaxis+1)] - [x,y])**2,1).argmin() # find nearest point
      x,y = corr[n,0:2]
      ax1.plot(x,y,'+',ms=10,mew=2,color='blue',scalex=False,scaley=False)
      x,y = corr[n,2:4]
      ax2.plot(x,y,'+',ms=10,mew=2,color='red',scalex=False,scaley=False)
      # we redraw only the concerned axes
      renderer = fig.canvas.get_renderer()
      ax1.draw(renderer)  
      ax2.draw(renderer)
      fig.canvas.blit(ax1.bbox)
      fig.canvas.blit(ax2.bbox)
    
    def noticks():
      xticks([])
      yticks([])
    clf()
    ax1 = subplot(221)
    ax1.numaxis = 0
    imshow(img0,interpolation='nearest')
    noticks()
    ax2 = subplot(222)
    ax2.numaxis = 1
    imshow(img1,interpolation='nearest')
    noticks()
    
    ax = subplot(223)
    ax.numaxis = -1
    imshow(img0,interpolation='nearest')
    for m in set_max:
      plot(corr[corr[:,5]==m,0],corr[corr[:,5]==m,1],'+',ms=10,mew=2,color=colors[m],scalex=0,scaley=0)
    noticks()
    
    ax = subplot(224)
    ax.numaxis = -1
    imshow(img1,interpolation='nearest')
    for m in set_max:
      plot(corr[corr[:,5]==m,2],corr[corr[:,5]==m,3],'+',ms=10,mew=2,color=colors[m],scalex=0,scaley=0)
    noticks()
    show()
    subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                    wspace=0.02, hspace=0.02)
    
    fig = get_current_fig_manager().canvas.figure
    cid_move = fig.canvas.mpl_connect('motion_notify_event',motion_notify_callback)

    show()
    fig.canvas.mpl_disconnect(cid_move)



if __name__=='__main__':
  args = sys.argv[1:]
  img0 = array(Image.open(args[0]).convert('RGB'))
  img1 = array(Image.open(args[1]).convert('RGB'))
  
  retained_matches = []
  for line in sys.stdin:
    line = line.split()
    if not line or len(line)!=6 or not line[0][0].isdigit():  continue
    x0, y0, x1, y1, score, index = line
    retained_matches.append((float(x0),float(y0),float(x1),float(y1),float(score),float(index)))
  
  assert retained_matches, 'error: no matches piped to this program'
  show_correspondences(img0, img1, array(retained_matches))
  


























