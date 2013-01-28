import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

''' colormap '''
colorlist = np.asarray([
    [0,0,0,0],
    [160,212,205,255],
    [164,154,219,255],
    [0,0,255,255],
    [24,203,255,255],
    [231,255,0,255],
    [231, 66, 0, 255],
    [226,143,122,255],
    [228,0,145,255],
    [226,164,43,255],
    [255,0,0,255],
    [255,151,0,255],
    [206,73,255,255],
    [142,5,10,255],
    [18,95,0,255],
    [8,212,125,255],
    [0,255,59,255],
    ])/255.1
from matplotlib.colors import ListedColormap
mycmap = ListedColormap(colorlist[:,:3], 'segmentation')
mycmap.set_bad(color=(0,0,0,0))

def plot_cross(gray, sol):
    figsize = np.asarray(gray.shape)*(10.0/gray.shape[0])
    fig = pyplot.figure(figsize=figsize)
    pyplot.imshow(gray, interpolation='nearest', cmap='gray', figure=fig)
   
    mseg = np.ma.masked_array(sol, mask=(sol==0), fill_value=0)
    pyplot.imshow(mseg, interpolation='nearest', cmap=mycmap, vmin=0, vmax=16, alpha=0.5, figure=fig) 

    pyplot.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #print 'done cross'
    return fig

if __name__=='__main__':
    from rwsegment import io_analyze
    import glob
    
    if len(sys.argv)==1: sys.exit()

    ## find image name
    name_index = sys.argv.index('-n')
    name = sys.argv[name_index+1]
    if name[0]=='f':
       foldname = name
       imname = name[4:]

    ## slice index
    islice_index = sys.argv.index('-i')
    islice = int(sys.argv[islice_index + 1])

    ## load images
    impath = '/workdir/baudinpy/01_register/'

    method_index = sys.argv.index('-m')
    method = sys.argv[method_index+1]
    methods = {
        'svm': '/workdir/baudinpy/segmentation_out/segmentation/2012.12.18*/svm*/',
        'handtune': '/workdir/baudinpy/**/segmentation/2012.12.18*/entropy*',
        }
    segpath = methods[method] 
    print segpath
    
    im = io_analyze.load('{}/{}/gray.hdr'.format(impath,imname))
    seg = io_analyze.load(glob.glob('{}/{}/sol.hdr'.format(segpath,foldname))[0])

    ## plot cross sections
    fig = plot_cross(im[islice], seg[islice])
    outdir = '/home/baudinpy/plots/'
    filename = 'cross_{}_{}_{}'.format(method,foldname, islice)
    print 'filename={}'.format(filename)
    pyplot.savefig('{}/{}.png'.format(outdir,filename))
    
