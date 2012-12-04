import os
import numpy as np
import plot_utils
reload(plot_utils)
from matplotlib import pyplot

def get_dices_from_path(path, filename='dice.txt'):

    def func(args, dirname, fnames):
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
        {'name':'Constant 1e-2', 'path':'segmentation/2012.11.30.test/constant1e-2/'},
        {'name':'Entropy 1e-2',  'path':'segmentation/2012.11.30.test/entropy1e-2/'},
        ]
    
    for i in range(len(methods)):
        methods[i]['values'] = get_dices_from_path(methods[i]['path'])
    
fig = plot_utils.plot_dice_labels(methods, labelset=[13,14,15,16], perlabel=True)

pyplot.show()