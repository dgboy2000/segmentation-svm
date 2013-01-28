import os
import sys
import numpy as np
import glob
import plot_utils
import platform
reload(plot_utils)
from matplotlib import pyplot

def get_values(method, filename='dice.txt'):
    def func(values, dirname, fnames):
        print dirname
        if filename in fnames:
            file = '{}/{}'.format(dirname, filename)
            f = open(file, 'r')
            for line in f.readlines():
                key, value = line.split()
                if key.isdigit():
                    key = int(key)
                value = float(value)
                if not key in values: values[key] = []
                else: values[key].append(value)
            #list = np.loadtxt(file)
            #labels = list[:,0]
            #values = list[:,1]
            #for label, value in zip(labels, values):
            #    if not dices.has_key(label): dices[label] = []
            #    dices[label].append(value)

    paths = glob.glob(method['path']) 
    values = {}
    for path in paths:
        os.path.walk(path, func, values)
    method['values'] = values    


if __name__=='__main__':
    
  
    ## handtuned
    method_enty1e0 = {
        'name':'enty1e0','title':'Entropy 1e0', 
        'path':'/workdir/baudinpy/segmentation_out/segmentation/2012.12.18.segmentation/entropy1e0/*', 
        'x':'',
        }
    method_const1e_2 = {
        'name':'cst1e-2','title':'Constant 1e-2', 
        'path':'/workdir/baudinpy/segmentation_out/segmentation/2012.12.18.segmentation/constant1e-2/*', 
        'x':'',
        }
    method_enty1e_2 = {
        'name':'enty1e-2','title':'Entropy 1e-2', 
        'path':'/workdir/baudinpy/segmentation_out/segmentation/2012.12.18.segmentation/entropy1e-2/*', 
        'x':'',
        }
    method_entyn1e_2 = {
        'name':'enty21e-2','title':'Entropy2 1e-2', 
        'path':'/workdir/baudinpy/segmentation_out/segmentation/2012.12.18.segmentation/entropyn1e-2/*', 
        'x':'',
        }
    method_const1e_2_I1e_2= {
        'name':'const1e-2_I1e-2','title':'Constant 1e-2 & Intensity 1e-2',
        'path':'/workdir/baudinpy/segmentation_out/segmentation/2012.12.18.segmentation/constant1e-2Intensity1e-2/*', 
        'x':'',
        }

    methods = [method_enty1e0, method_const1e_2, method_enty1e_2, method_entyn1e_2, method_const1e_2_I1e_2]

    get_values(method_enty1e0, filename='dice.txt')
    get_values(method_const1e_2, filename='dice.txt')
    get_values(method_enty1e_2, filename='dice.txt')
    get_values(method_entyn1e_2, filename='dice.txt')
    get_values(method_const1e_2_I1e_2, filename='dice.txt')

    ## make plot
    fig = plot_utils.plot_dice_labels(methods, labelset=[13,14,15,16], perlabel=True, ratio=0.5)
    
    ## save
    if platform.system()=='Windows': 
        pyplot.show()
    if '-s' in sys.argv:
        outdir = '../plots/'
        fname = 'dice_{}'.format('_'.join([m['name'] for m in methods]))
        print 'saving in file: {}'.format(fname)
        pyplot.savefig('{}{}.png'.format(outdir, fname))
    
