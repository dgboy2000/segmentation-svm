import os
import numpy as np
from scipy import ndimage

from rwsegment import ioanalyze
from rwsegment import rwmean_svm

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
    
    if (not force_recompute) and os.path.exists(file_prior):
        # logger.info('load prior')
        mask  = ioanalyze.load(file_mask)
        prior = np.load(file_prior)
    else:
        # logger.info('compute prior')
        generator = rwmean_svm.PriorGenerator(labelset)
        
        for train in config.vols:
            if test==train: continue
            file_seg = config.dir_reg + test + train + 'regseg.hdr'
            seg = ioanalyze.load(file_seg)
            generator.add_training_data(seg)
        
        from scipy import ndimage
        mask    = generator.get_mask()
        struct  = np.ones((7,)*mask.ndim)
        mask    = ndimage.binary_dilation(
                mask.astype(bool),
                structure=struct,
                )
        prior = generator.get_prior(mask=mask)
        
        nlabel = len(labelset)
        segprior = np.zeros(seg.shape)
        segprior[mask] = labelset[
            np.argmax(prior['mean'].reshape((-1,nlabel),order='C'),axis=1)]
            
        entropymap = np.zeros(seg.shape)
        entropymap[mask] = \
            prior['entropy'].reshape((-1,nlabel),order='C')[:,0]
        entropymap = entropymap / np.log(nlabel) * 2**15
            
        
        np.savez(file_prior,**prior)
        ioanalyze.save(file_mask, mask.astype(np.int32))
        ioanalyze.save(file_segprior, segprior.astype(np.int32))
        ioanalyze.save(file_entropymap, entropymap.astype(np.int32))
        
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