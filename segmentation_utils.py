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
    
    if not force_recompute or os.path.exists(file_prior):
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
        
        np.savez(file_prior,**prior)
        ioanalyze.save(file_mask, mask.astype(np.int32))
        
    return prior, mask