import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import config
labelset = [13,14,15,16]
muscles = ['rectus\nfemoris', 'vastus\nlateralis', 'vastus\nintermedius', 'vastus\nmedialis']

dir_results = '/workdir/baudinpy/segmentation_out/segmentation/2012.09.10.segmentation_all//'
methods = {
    'cst1': ['Constant 1e-2', 'constant1e-2'],
    'enty1': ['Entropy 1e-1', 'entropy1e-1'],
    'enty2': ['Entropy 1e-2', 'entropy1e-2'],
    'entycst1': ['Entropy 1e-2 / Intensity 1e-2', 'entropy1e-2_intensity1e-2'],
    'entycst2': ['Entropy 1e-3 / Intensity 1e-2', 'entropy1e-3_intensity1e-2'],
   }

vols = config.vols.keys()

class GetDices(object):
    def __init__(self, method='enty1'):
        self.method = method
        self.title, self.dir = methods[method]
        self.dices = {'title': self.title}
    def __call__(self, args, dirnames, fnames):
        dir = self.dir
        if 'dice.txt' in fnames:
            path = os.path.normpath(dirnames).split(os.sep)
            if dir in path:
                dice = np.loadtxt(dirnames + '/dice.txt')
                labels, coefs = dice[:,0].astype(int), dice[:,1]
                for label, coef in zip(labels, coefs):
                    if not self.dices.has_key(label): self.dices[label] = []
                    self.dices[label].append(coef)

colors = [r'blue', r'green', r'red', r'yellow', r'purple', r'orange']

def plot_dices(dices_list, all=True):
    fig = pyplot.figure()
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
            print [-0.4 + (i+0.5)*width],
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
    
    pyplot.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.05)
    pyplot.title(r'Dice coefficients')
    pyplot.yticks(range(nset), yticks)
    from matplotlib.ticker import MultipleLocator
    minorLocator   = MultipleLocator(0.05)
    ax1.xaxis.set_minor_locator(minorLocator) 
    ax1.xaxis.grid(True, linestyle='-', which='both',color='lightgrey', alpha=0.9)

    pyplot.axis([0.5,1,nset -  0.5,-0.5])
    pyplot.legend(legends_box, legends, loc=3)

dices_list = []
for method in ['cst1', 'enty2', 'entycst1']:
    get_dices = GetDices(method=method)
    os.path.walk(dir_results, get_dices, {})
    dices = get_dices.dices
    dices_list.append(dices)

box = plot_dices(dices_list, all=False)
pyplot.savefig('test.png')

