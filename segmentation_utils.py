import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment_prior
reload(rwsegment_prior)

import config
reload(config)


def load_or_compute_prior_and_mask(test, force_recompute=False):

    labelset = np.asarray(config.labelset)
    outdir = 'prior/' + test
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    ## load mask and prior
    prior = None
    file_mask  = outdir + 'mask.hdr'
    file_prior = outdir + 'prior.npz'
    file_segprior = outdir + 'segprior.hdr'
    file_entropymap = outdir + 'entropymap.hdr'
    
    if (not force_recompute) and os.path.exists(file_prior):        # logger.info('load prior')
        mask  = io_analyze.load(file_mask)
        prior = np.load(file_prior)
    else:
        generator = rwsegment_prior.PriorGenerator(labelset)
        
        for train in config.vols:
            if test==train: continue
            logger.debug('load training img: {}'.format(train))
            
            file_seg = config.dir_reg + test + train + 'regseg.hdr'
            seg = io_analyze.load(file_seg)
            generator.add_training_data(seg)
        
        from scipy import ndimage
        mask    = generator.get_mask()
        struct  = np.ones((7,)*mask.ndim)
        mask    = ndimage.binary_dilation(
                mask.astype(bool),
                structure=struct,
                )
        prior = generator.get_prior(mask)
        
        nlabel = len(labelset)
        segprior = np.zeros(mask.shape)
        segprior[mask] = labelset[np.argmax(prior['mean'],axis=0)]
            
        entropymap = np.zeros(mask.shape)
        entropymap[mask] = np.sum(
            np.log(prior['mean'] + 1e-10)*prior['mean'],
            axis=0)
        entropymap = entropymap / np.log(nlabel) * 2**15
            
        np.savez(file_prior,**prior)
        
        io_analyze.save(file_mask, mask.astype(np.int32))
        io_analyze.save(file_segprior, segprior.astype(np.int32))
        io_analyze.save(file_entropymap, entropymap.astype(np.int32))
        
    return prior, mask
    
    
def compute_dice_coef(seg1, seg2, labelset=None):
    if labelset is None:
        lbset = np.union(np.unique(seg1), np.unique(seg2))
    else:
        lbset = np.asarray(labelset, dtype=int)
    
    seg1.flat[~np.in1d(seg1, lbset)] = -1
    seg2.flat[~np.in1d(seg2, lbset)] = -1
    
    dicecoef = {}
    for label in lbset:
        l1 = (seg1==label)
        l2 = (seg2==label)
        d = 2*np.sum(l1&l2)/(1e-9 + np.sum(l1) + np.sum(l2))
        dicecoef[label] = d
    return dicecoef
    
from rwsegment import utils_logging
logger = utils_logging.get_logger('logger_utils',utils_logging.DEBUG)