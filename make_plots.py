import os
import numpy as np
import plot_utils
import platform
reload(plot_utils)
from matplotlib import pyplot

def get_dices_from_path(path, filename='dice.txt'):
    def func(args, dirname, fnames):
        print dirname
        if filename in fnames:
            file = '{}/{}'.format(dirname, filename)
            list = np.loadtxt(file)
            labels = list[:,0].astype(int)
            values = list[:,1]
            for label, value in zip(labels, values):
                if not dices.has_key(label): dices[label] = []
                dices[label].append(value)
    dices = {}        
    os.path.walk(path, func, dices)
    return dices


if __name__=='__main__':
    
    methods = [
        #{'name': 'test', 'title':'Test', 'path':'/home/baudinpy/code_temp/f01/'},
        #{'name': 'const', 'title':'Constant', 'path':'segmentation/2012.11.30.test/constant1e-2/', 'x':1},
        #{'name': 'entro', 'title':'Entropy',  'path':'segmentation/2012.11.30.test/entropy1e-2/', 'x':2},
        ]
        
    path_l = '/workdir/baudinpy/segmentation_out/learning/'
    methods_lLs = [{'name':'lLsC{}'.format(c), 'path':path_l+'2012.12.04.exp_latent_DACI_crop2_Lsdloss_x1000_C{}'.format(c), 'x':c}\
        for c in [1e-2, 1e0, 1e2, 1e4]]
    methods_bLs = [{'name':'bLsC{}'.format(c), 'path':path_l+'2012.12.04.exp_baseline_crop10_Lsdloss_x1000_C{}'.format(c), 'x':c}\
        for c in [1e-2, 1e0, 1e2, 1e4]]
    methods_lLnC1e0 = [{'name':'lLnCp{}C{}'.format(cp,c), 'path':path_l+'2012.12.04.exp_latent_DACI_crop2_Lnone_x1000_Cp{}_C{}'.format(cp,c), 'x':cp}\
        for c in [1e0] for cp in  [1e-2, 1e2, 1e6, 1e10]]
    methods_bLnC1e0 = [{'name':'bLnCp{}C{}'.format(cp,c), 'path':path_l+'2012.12.04.exp_baseline_crop10_Lnone_x1000_Cp{}_C{}'.format(cp,c), 'x':cp}\
        for c in [1e0] for cp in  [1e-2, 1e2, 1e6, 1e10]]
    
    series = [
        #{'name': 'test', 'title': 'Test series', 'methods':methods}
        {'name':'bLs', 'title': 'Baseline sdloss crop10', 'methods':methods_bLs},
        {'name':'lLs', 'title': 'Latent sdloss aACI crop2', 'methods':methods_lLs},
        {'name':'bLnC1e0', 'title': 'Baseline crop10 C=1e0', 'methods':methods_bLnC1e0},
        {'name':'lLnC1e0', 'title': 'Latent aACI crop2 C=1e0', 'methods':methods_lLnC1e0},
        ]
    #import ipdb; ipdb.set_trace()
    
    ## get dices
    for i in range(len(methods)):
        #methods[i]['values'] = get_dices_from_path(methods[i]['path'], filename='dice.txt')
        methods_lLs[i]['values'] = get_dices_from_path(methods_lLs[i]['path'], filename='dice.txt')
        methods_bLs[i]['values'] = get_dices_from_path(methods_bLs[i]['path'], filename='dice.txt')
        methods_lLnC1e0[i]['values'] = get_dices_from_path(methods_lLnC1e0[i]['path'], filename='dice.txt')
        methods_bLnC1e0[i]['values'] = get_dices_from_path(methods_bLnC1e0[i]['path'], filename='dice.txt')
    
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
        fname = 'dice_' + '_'.join([m['name'] for m in methods])
        pyplot.savefig('{}{}.png'.format(outdir, fname))
    