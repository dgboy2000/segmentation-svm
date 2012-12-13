import os
import numpy as np
import glob
import plot_utils
import platform
reload(plot_utils)
from matplotlib import pyplot

def get_dices(method, filename='dice.txt'):
    def func(dices, dirname, fnames):
        print dirname
        if filename in fnames:
            file = '{}/{}'.format(dirname, filename)
            list = np.loadtxt(file)
            labels = list[:,0].astype(int)
            values = list[:,1]
            for label, value in zip(labels, values):
                if not dices.has_key(label): dices[label] = []
                dices[label].append(value)

    paths = glob.glob(method['path']) 
    dices = {}
    for path in paths:
        os.path.walk(path, func, dices)
    method['values'] = dices    


if __name__=='__main__':
    
    methods = [
        #{'name': 'test', 'title':'Test', 'path':'/home/baudinpy/code_temp/f01/'},
        #{'name': 'const', 'title':'Constant', 'path':'segmentation/2012.11.30.test/constant1e-2/', 'x':1},
        #{'name': 'entro', 'title':'Entropy',  'path':'segmentation/2012.11.30.test/entropy1e-2/', 'x':2},
        ]
        
    path_l = '/workdir/baudinpy/segmentation_out/learning/'
    methods_lLs = [{'name':'lLsC{}'.format(c), 'path':path_l+'2012.12.06.exp_latent_DACI_crop2_Lsdloss_x1000_C{}'.format(c), 'x':c}\
        for c in [1e0, 1e2, 1e3, 1e4]]
    methods_bLs = [{'name':'bLsC{}'.format(c), 'path':path_l+'2012.12.64.exp_baseline_crop10_Lsdloss_x1000_C{}'.format(c), 'x':c}\
        for c in [1e0, 1e2, 1e3, 1e4]]

    methods_lLnC1e0 = [{'name':'lLnCp{}C{}'.format(cp,c), 'path':path_l+'2012.12.06.exp_latent_DACI_crop2_Lnone_x1000_Cp{}_C{}'.format(cp,c), 'x':cp}\
        for c in [1e0] for cp in  [1e-2, 1e0, 1e2, 1e6]]
    methods_lLnC1e2 = [{'name':'lLnCp{}C{}'.format(cp,c), 'path':path_l+'2012.12.06.exp_latent_DACI_crop2_Lnone_x1000_Cp{}_C{}'.format(cp,c), 'x':cp}\
        for c in [1e2] for cp in  [1e-2, 1e0, 1e2, 1e6]]
    methods_lLnC1e3 = [{'name':'lLnCp{}C{}'.format(cp,c), 'path':path_l+'2012.12.06.exp_latent_DACI_crop2_Lnone_x1000_Cp{}_C{}'.format(cp,c), 'x':cp}\
        for c in [1e3] for cp in  [1e-2, 1e0, 1e2, 1e6]]
    methods_lLnC1e4 = [{'name':'lLnCp{}C{}'.format(cp,c), 'path':path_l+'2012.12.06.exp_latent_DACI_crop2_Lnone_x1000_Cp{}_C{}'.format(cp,c), 'x':cp}\
        for c in [1e4] for cp in  [1e-2, 1e0, 1e2, 1e6]]

    methods_bLnC1e0 = [{'name':'bLnCp{}C{}'.format(cp,c), 'path':path_l+'2012.12.04.exp_baseline_crop10_Lnone_x1000_Cp{}_C{}'.format(cp,c), 'x':cp}\
        for c in [1e0] for cp in  [1e-2, 1e2, 1e6, 1e10]]
   
    ## handtuned
    method_enty1e_2 = {'name':'enty1e_2', 'path':'/workdir/baudinpy/segmentation_out/segmentation/2012.11.19.segmentation/entropy0.01/f*', 'x':''} 
    method_enty1e0 = {'name':'enty1e0', 'path':'/workdir/baudinpy/segmentation_out/segmentation/2012.11.19.segmentation/entropy1.0/f*', 'x':''} 

    series = [
        #{'name': 'test', 'title': 'Test series', 'methods':methods}
        #{'name':'bLs', 'title': 'Baseline sdloss crop10', 'methods':methods_bLs},
        #{'name':'lLs', 'title': 'Latent sdloss aACI crop2', 'methods':methods_lLs},
        #{'name':'bLnC1e0', 'title': 'Baseline crop10 C=1e0', 'methods':methods_bLnC1e0},
        #{'name':'lLnC1e0', 'title': 'Latent aACI crop2 C=1e0', 'methods':methods_lLnC1e0},
        {'name':'lLnC1e0', 'title': 'Latent aACI crop2 C=1e0', 'methods':methods_lLnC1e0},
        {'name':'lLnC1e2', 'title': 'Latent aACI crop2 C=1e2', 'methods':methods_lLnC1e2},
        {'name':'lLnC1e3', 'title': 'Latent aACI crop2 C=1e3', 'methods':methods_lLnC1e3},
        {'name':'lLnC1e4', 'title': 'Latent aACI crop2 C=1e4', 'methods':methods_lLnC1e4},
        {'name':'handtuned1', 'title': 'Handtuned (best)', 'methods':[method_enty1e_2]},
        {'name':'handtuned2', 'title': 'Handtuned (init)', 'methods':[method_enty1e0]},
        ]

    ## get dices
    #for i in range(len(methods)):
        #methods[i]['values'] = get_dices_from_path(methods[i]['path'], filename='dice.txt')
    #[get_dices(method, filename='dice.txt') for method in  methods_lLs]
    #[get_dices(method, filename='dice.txt') for method in  methods_bLs]
    #[get_dices(method, filename='dice.txt') for method in  methods_lLnC1e0]
    #[get_dices(method, filename='dice.txt') for method in  methods_bLnC1e_2]
    [get_dices(method, filename='dice.txt') for method in  methods_lLnC1e0]
    [get_dices(method, filename='dice.txt') for method in  methods_lLnC1e2]
    [get_dices(method, filename='dice.txt') for method in  methods_lLnC1e3]
    [get_dices(method, filename='dice.txt') for method in  methods_lLnC1e4]
    get_dices(method_enty1e_2, filename='dice.txt')
    get_dices(method_enty1e0, filename='dice.txt')

    ## make plot
    #fig = plot_utils.plot_dice_labels(methods, labelset=[13,14,15,16], perlabel=False)
    fig = plot_utils.plot_dice_series(series, labelset=[13,14,15,16])
    
    ## save
    if platform.system()=='Windows': 
        #outdir = './'
        print 'blah'
        pyplot.show()
    else:
        outdir = '/home/baudinpy/plots/'    
        #fname = 'dice_' + '_'.join([m['name'] for m in methods])
        fname = 'dice_series_' + '_'.join([m['name'] for m in series])
        #fname = 'test'
        print fname
        pyplot.savefig('{}{}.png'.format(outdir, fname))
    
