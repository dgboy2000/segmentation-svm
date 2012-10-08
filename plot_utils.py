import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import config
labelset = [13,14,15,16]
muscles = ['Rectus\nfemoris', 'vastus\nlateralis', 'vastus\nintermedius', 'vastus\nmedialis']
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

vols = config.vols.keys()

class GetDices(object):
    def __init__(self, dir, title):
        self.dir = dir
        self.title = title
        self.dices = {'title': self.title}
        self.nsample = 0
    def __call__(self, args, dirnames, fnames):
        dir = self.dir
        file_dice = args #args.pop('file_dice','dice_labels.txt')
        if file_dice in fnames:
            path = os.path.normpath(dirnames).split(os.sep)
            if dir in path:
                #print 'loading file {}'.format(dirnames + '/' + file_dice)
                dice = np.loadtxt(dirnames + '/' + file_dice)
                labels, coefs = dice[:,0].astype(int), dice[:,1]
                for label, coef in zip(labels, coefs):
                    if not self.dices.has_key(label): self.dices[label] = []
                    self.dices[label].append(coef)
                self.nsample += 1
            self.dices['nsample'] = self.nsample


colors = [r'blue', r'green', r'red', r'yellow', r'purple', r'orange']

def plot_dices(dices_list, all=True, ratio=1):
    figsize = [10,10]
    figsize[1] = int(ratio*figsize[0])
    print figsize
    fig = pyplot.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    nbox = len(dices_list)
    width = 0.8/(nbox) 
    legends   = []
    legends_box   = []

    if all:
        set = [labelset]
        yticks = ['All muscles']
    else:
        set = [[i] for i in labelset]
        yticks = [muscles[i] for i in range(len(labelset))]
        
    nset = len(set)
    for iset, s in enumerate(set):
        for i, dices in enumerate(dices_list):
            values = []
            for label in dices:
                if label in s:
                    values.extend(dices[label])
            legend = dices['title']
            print 'nsample:',dices['nsample']
            #print [-0.4 + (i+0.5)*width],
            bp = pyplot.boxplot(
                [values],
                positions = [iset + -0.4 + (i+0.5)*width],
                widths    = [width],
                vert=0,
                whis=1.5,
                notch=1,
                bootstrap=10000,
                )
            pyplot.setp(bp['boxes'],    color=colors[i], zorder=0)
            pyplot.setp(bp['medians'],  color=colors[i])
            pyplot.setp(bp['whiskers'], color='black')
            pyplot.setp(bp['fliers'],   color='red', marker='+')

            for box in bp['boxes']:
                pyplot.fill(box.get_xdata(), box.get_ydata(), color=colors[i], alpha=0.3)
            
            if iset==0:
                legends.append(legend)
                r = pyplot.Rectangle((0,0),0,0,facecolor=colors[i], edgecolor=colors[i],alpha=0.3)
                legends_box.append(r)
    
    pyplot.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    pyplot.title(r'$\mathrm{Dice\ coefficients}$')
    pyplot.yticks(range(nset), yticks, rotation=90)
    from matplotlib.ticker import MultipleLocator
    minorLocator   = MultipleLocator(0.05)
    ax1.xaxis.set_minor_locator(minorLocator) 
    ax1.xaxis.grid(True, linestyle='-', which='both',color='lightgrey', alpha=0.9)

    pyplot.axis([0.5,1,nset -  0.5,-0.5])
    pyplot.legend(legends_box, legends, loc=3, prop={'size':12})
    return fig


def plot_dices_per_slice(dices_list):
    fig = pyplot.figure()
    ax1 = fig.add_subplot(111)
    nbox = len(dices_list)
    width = 0.8/(nbox) 
    legends   = []
    legends_box   = []

    for i, dices in enumerate(dices_list):
        legend = dices['title']
        print 'nsample:',dices['nsample']
        left = []
        vals = []
        err = []
        minv = []
        maxv = []
        for l in dices.keys():
            if str(l).isdigit():
                n = len(dices[l])
                if n == 0:
                    minv.append(0)
                    maxv.append(0)
                    vals.append(0)
                elif n<=4:  
                    minv.append(0)
                    maxv.append(np.mean(dices[l]))
                    vals.append(np.mean(dices[l]))
                else:
                    v = np.sort(dices[l]) 
                    vals.append(np.median(dices[l]))
                    minv.append(np.mean(v[:n/4]))
                    maxv.append(np.mean(v[(n*3)/4:]))
                left.append(l)
        bp = pyplot.barh(
            np.array(left),
            np.array(maxv),
            height=[width for w in left],
            color=(0,0,1),
            alpha=0.3,
            )
       
        bp = pyplot.barh(
            np.array(left),
            np.array(vals),
            height=[width for w in left],
            color=(0,0,1),
            alpha=0.8,
            )
        bp = pyplot.barh(
            np.array(left),
            np.array(minv),
            height=[width for w in left],
            color=(0,0,0.5),
            alpha=1,
            )
        legends.append(legend)
        r = pyplot.Rectangle((0,0),0,0,facecolor=colors[i], edgecolor=colors[i],alpha=0.3)
        legends_box.append(r)
   
    pyplot.xlabel(r'$\mathrm{slices}$') 
    pyplot.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    pyplot.title(r'$\mathrm{Dice\ coefficients\ per\ slice\ for\ method:' + legend + '}$')

    #pyplot.yticks(range(nset), yticks)
    from matplotlib.ticker import MultipleLocator
    minorLocator   = MultipleLocator(0.05)
    ax1.xaxis.set_minor_locator(minorLocator) 
    ax1.xaxis.grid(True, linestyle='-', which='both',color='lightgrey', alpha=0.9)


    #pyplot.axis([0.5,1,nset -  0.5,-0.5])
    #pyplot.legend(legends_box, legends, loc=3)


def plot_cross(gray, sol, seg):
    figsize = np.asarray(gray.shape)*(10.0/gray.shape[0])
    pyplot.figure(figsize=figsize)
    pyplot.imshow(gray, interpolation='nearest', cmap='gray')
   
    mseg = np.ma.masked_array(sol, mask=(sol==0), fill_value=0)
    pyplot.imshow(mseg, interpolation='nearest', cmap=mycmap, vmin=0, vmax=16, alpha=0.5) 

    pyplot.subplots_adjust(left=0, right=1, top=1, bottom=0)
    print 'done cross'

if __name__=='__main__':
    import sys

    dir1 = '/workdir/baudinpy/segmentation_out/segmentation/2012.09.10.segmentation_all//'
    dir2 = '/workdir/baudinpy/segmentation_out/segmentation/2012.09.25.segmentation_variance_allm//'
    dir3 = '/workdir/baudinpy/segmentation_out/segmentation/2012.09.26.segmentation_variance_allm//'
    
    methods = {
        'cst1': ['Constant 1e-2', 'constant1e-2', dir1],
        'enty1': ['Entropy 1e-1', 'entropy1e-1', dir1],
        'enty2': ['Entropy 1e-2', 'entropy1e-2', dir1],
        'entyint1': ['Entropy 1e-2 / Intensity 1e-2', 'entropy1e-2_intensity1e-2', dir1],
        'entyint2': ['Entropy 1e-3 / Intensity 1e-2', 'entropy1e-3_intensity1e-2', dir1],
        'var0': ['Variance', 'variance1e-0', dir2],
        'var1': ['Variance', 'variance1e-1', dir2],
        'varcmap0': ['Variance + CMap', 'variancecmap1e-0', dir2],
        'varcmap1': ['Variance + CMap', 'variancecmap1e-1', dir2],
        'mean': ['Registration', 'mean', dir3],
       }

    if '--dice' in sys.argv:
        dices_list = []
        #selected_methods = ['cst1', 'enty2', 'entyint1']
        selected_methods = ['mean', 'var1','varcmap1']
        #selected_methods = ['mean','var1', 'varcmap1']
        for method in selected_methods:
            get_dices = GetDices(methods[method][1], methods[method][0])
            os.path.walk(methods[method][2], get_dices, 'dice_labels.txt')
            dices = get_dices.dices
            dices_list.append(dices)
        
            get_dices2 = GetDices(methods[method][1], methods[method][0])
            os.path.walk(methods[method][2], get_dices2, 'dice_slices.txt')
            dices2 = get_dices2.dices
        
            #bar = plot_dices_per_slice([dices2])
            #pyplot.savefig('../plots/dices_slice_{}.png'.format(method), dpi=300)
        
        box = plot_dices(dices_list, all=False)
        strmethods = '-'.join(selected_methods)
        pyplot.savefig('/home/baudinpy/plots/dices_{}.png'.format(strmethods), dpi=300)
        box_all = plot_dices(dices_list, all=True, ratio=0.4)
        pyplot.savefig('/home/baudinpy/plots/dices_{}_all.png'.format(strmethods), dpi=300)


    if '--cross' in sys.argv:
        method = 'enty2'
        test = '01/'
        islice = 40
        use_gt = False
        if '--gt' in sys.argv:
           use_gt = True
        if '--method' in sys.argv:
            i = sys.argv.index('--method')
            method = sys.argv[i+1]
        if '--slice' in sys.argv:
            i = sys.argv.index('--slice')
            islice = int(sys.argv[i+1])
        if '--test' in sys.argv:
            i = sys.argv.index('--test')
            test = sys.argv[i+1] + '/'
        print sys.argv


        from rwsegment import io_analyze
        dir = methods[method][2] + methods[method][1] + '/'
        print dir
        print test
        print islice
        sol = io_analyze.load(dir + test + 'sol.hdr')[islice]
        seg = io_analyze.load(config.dir_reg + test + 'seg.hdr')[islice]
        gray = io_analyze.load(config.dir_reg + test + 'gray.hdr')[islice]
        if use_gt:
            method = 'gt'
            plot_cross(gray, seg, seg)
        else:
            plot_cross(gray, sol, seg)
        pyplot.savefig('/home/baudinpy/plots/cross_{}_{}_{}.png'.format(method, test[:-1], islice))
        print 'saving to: {}'.format('/home/baudinpy/plots/cross_{}_{}_{}.png'.format(method, test[:-1], islice))

