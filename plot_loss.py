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
    
    path_l = '/workdir/baudinpy/segmentation_out/learning/'
    
    methods_lLnC1e_1 = [
        {'name':'lLnCp{}C{}'.format(cp,c), 
         'path':path_l+'2012.12.13.exp_latent_DDACI_crop2_Lnone_x1000_Cp{}_C{}/**/*'.format(cp,c), 
         'x':'{:.0e}'.format(cp)}\
        for c in [1e-1] for cp in  [1e-3, 1e-2, 1e0, 1e2, 1e4]]
    methods_lLnC1e0 = [
        {'name':'lLnCp{}C{}'.format(cp,c), 
         'path':path_l+'2012.12.13.exp_latent_DDACI_crop2_Lnone_x1000_Cp{}_C{}/**/*'.format(cp,c), 
         'x':'{:.0e}'.format(cp)}\
        for c in [1e0] for cp in  [1e-3, 1e-2, 1e0, 1e2, 1e4]]
    methods_lLnC1e1 = [
        {'name':'lLnCp{}C{}'.format(cp,c), 
         'path':path_l+'2012.12.13.exp_latent_DDACI_crop2_Lnone_x1000_Cp{}_C{}/**/*'.format(cp,c), 
         'x':'{:.0e}'.format(cp)}\
        for c in [1e1] for cp in  [1e-3, 1e-2, 1e0, 1e2, 1e4]]
    methods_lLnC1e2 = [
        {'name':'lLnCp{}C{}'.format(cp,c), 
         'path':path_l+'2012.12.13.exp_latent_DDACI_crop2_Lnone_x1000_Cp{}_C{}/**/*'.format(cp,c), 
         'x':'{:.0e}'.format(cp)}\
        for c in [1e2] for cp in  [1e-3, 1e-2, 1e0, 1e2, 1e4]]
    methods_lLnC1e3 = [
        {'name':'lLnCp{}C{}'.format(cp,c), 
         'path':path_l+'2012.12.13.exp_latent_DDACI_crop2_Lnone_x1000_Cp{}_C{}/**/*'.format(cp,c), 
         'x':'{:.0}'.format(cp)}\
        for c in [1e3] for cp in  [1e-3, 1e-2, 1e0, 1e2, 1e4]]

    methods_bLnC1e_1 = [
        {'name':'bLn{}C{}'.format(cp,c), 
         'path':path_l+'2012.12.13.exp_baseline_crop2_Lnone_x1000_Cp{}_C{}/**/*'.format(cp,c), 
         'x':'{:.0e}'.format(cp)}\
        for c in [1e-1] for cp in  [1e-3, 1e-2, 1e0, 1e2, 1e4]]
    methods_bLnC1e0 = [
        {'name':'bLn{}C{}'.format(cp,c), 
         'path':path_l+'2012.12.13.exp_baseline_crop2_Lnone_x1000_Cp{}_C{}/**/*'.format(cp,c), 
         'x':'{:.0e}'.format(cp)}\
        for c in [1e0] for cp in  [1e-3, 1e-2, 1e0, 1e2, 1e4]]
    methods_bLnC1e1 = [
        {'name':'bLn{}C{}'.format(cp,c), 
         'path':path_l+'2012.12.13.exp_baseline_crop2_Lnone_x1000_Cp{}_C{}/**/*'.format(cp,c), 
         'x':'{:.0e}'.format(cp)}\
        for c in [1e1] for cp in  [1e-3, 1e-2, 1e0, 1e2, 1e4]]
    methods_bLnC1e2 = [
        {'name':'bLn{}C{}'.format(cp,c), 
         'path':path_l+'2012.12.13.exp_baseline_crop2_Lnone_x1000_Cp{}_C{}/**/*'.format(cp,c), 
         'x':'{:.0e}'.format(cp)}\
        for c in [1e2] for cp in  [1e-3, 1e-2, 1e0, 1e2, 1e4]]
   
    ## handtuned
    method_enty1e0 = {
        'name':'enty1e0','title':'Hand-tuned', 
        'path':'/workdir/baudinpy/segmentation_out/segmentation/2012.12.18.segmentation/entropy1e0/f09*', 
        'x':'',
        }
    method_enty1e_2 = {
        'name':'enty1e-2','title':'Hand-tuned', 
        'path':'/workdir/baudinpy/segmentation_out/segmentation/2012.12.18.segmentation/entropy1e-2/f09*', 
        'x':'',
        }

    series1 = [
        {'name':'lLnC1e-1', 'title': r'Latent $\lambda=0.1$', 'methods':methods_lLnC1e_1},
        {'name':'bLnC1e-1', 'title': r'Baseline $\lambda=0.1$', 'methods':methods_bLnC1e_1},
        {'name':'handtuned2', 'title': 'Hand-tuned', 'methods':[method_enty1e0]*len(methods_lLnC1e0)},
        ]
    series2 = [
        {'name':'lLnC1e0', 'title': r'Latent $\lambda=1.0$', 'methods':methods_lLnC1e0},
        {'name':'bLnC1e0', 'title': r'Baseline $\lambda=1.0$', 'methods':methods_bLnC1e0},
        {'name':'handtuned2', 'title': 'Hand-tuned', 'methods':[method_enty1e0]*len(methods_lLnC1e0)},
        ]
    series3 = [       
        {'name':'lLnC1e1', 'title': r'Latent $\lambda=10.0$', 'methods':methods_lLnC1e1},
        {'name':'bLnC1e1', 'title': r'Baseline $\lambda=10.0$', 'methods':methods_bLnC1e1},
        {'name':'handtuned2', 'title': 'Hand-tuned', 'methods':[method_enty1e0]*len(methods_lLnC1e0)},
        ]   
    series4 = [
        {'name':'lLnC1e2', 'title': r'Latent $\lambda=100.0$', 'methods':methods_lLnC1e2},
        {'name':'bLnC1e2', 'title': r'Baseline $\lambda=100.0$', 'methods':methods_bLnC1e2},
        {'name':'handtuned2', 'title': 'Hand-tuned', 'methods':[method_enty1e0]*len(methods_lLnC1e0)},
        ]
        #{'name':'lLnC1e3', 'title': r'Latent $\lambda=1000.0$', 'methods':methods_lLnC1e3},
        #{'name':'bLnC1e2', 'title': r'baseline $\lambda=100.0$', 'methods':methods_bLnC1e2},
        #{'name':'handtuned2', 'title': 'Hand-tuned', 'methods':[method_enty1e0]*len(methods_lLnC1e0)},
        #]

    ## get dices
    [get_values(method, filename='losses.txt') for method in  methods_lLnC1e_1]
    [get_values(method, filename='losses.txt') for method in  methods_lLnC1e0]
    [get_values(method, filename='losses.txt') for method in  methods_lLnC1e1]
    [get_values(method, filename='losses.txt') for method in  methods_lLnC1e2]
    [get_values(method, filename='losses.txt') for method in  methods_lLnC1e3]
    [get_values(method, filename='losses.txt') for method in  methods_bLnC1e_1]
    [get_values(method, filename='losses.txt') for method in  methods_bLnC1e0]
    [get_values(method, filename='losses.txt') for method in  methods_bLnC1e1]
    [get_values(method, filename='losses.txt') for method in  methods_bLnC1e2]
    get_values(method_enty1e0, filename='losses.txt')
    get_values(method_enty1e_2, filename='losses.txt')

    ## make plot
    #fig = plot_utils.plot_dice_labels(methods, labelset=[13,14,15,16], perlabel=True, ratio=0.5)
    fig = plot_utils.plot_loss(series1, series2, series3, series4, ratio=0.5)
    
    ## save
    if platform.system()=='Windows': 
        #outdir = './'
        print 'blah'
        pyplot.show()
    else:
        outdir = '/home/baudinpy/plots/'
        if '-s' in sys.argv:    
            fname = 'dice_series_' + '_'.join([m['name'] for m in series])
        elif '-m' in sys.argv:
            fname = 'dice_' + '_'.join([m['name'] for m in methods])
        elif '-l' in sys.argv:
            fname = 'losses_' + '_'.join([m['name'] for m in series1])
        else:
            sys.exit()
        print fname
        pyplot.savefig('{}{}.png'.format(outdir, fname))
    
