import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment

import segmentation_utils 
reload(segmentation_utils)

from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
    
import config
reload(config)

from rwsegment import utils_logging
logger = utils_logging.get_logger('batch_rwpca',utils_logging.INFO)

class SegmentationBatch(object):
    
    def __init__(self):
        
        self.labelset  = np.asarray(config.labelset)
        self.force_recompute_prior = False
        
        self.params  = {
            'beta'             : 50,     # contrast parameter
            'return_arguments' :['image','y'],
            
            # optimization parameter
            'per_label': True,
            'optim_solver':'unconstrained',
            'rtol'      : 1e-6,
            'maxiter'   : 2e3,
            }
        
        
    def process_sample(self,test):

        ## get prior
        prior, mask = load_or_compute_pca_prior_and_mask(
            test,force_recompute=self.force_recompute_prior)
        seeds   = (-1)*mask
        
        ## load image
        file_name = config.dir_reg + test + 'gray.hdr'        
        logger.info('segmenting data: {}'.format(file_name))
        im      = io_analyze.load(file_name)
        file_gt = config.dir_reg + test + 'seg.hdr'
        seg     = io_analyze.load(file_gt)
        seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
        
           
        ## normalize image
        nim = im/np.std(im)
            
          
        ## start segmenting
        # import ipdb; ipdb.set_trace()
        sol,y = rwsegment_pca.segment(
            nim, 
            #anchor_api,
            seeds=seeds, 
            labelset=self.labelset, 
            **self.params
            )

        ## compute Dice coefficient per label
        dice    = compute_dice_coef(sol, seg,labelset=self.labelset)
        logger.info('Dice: {}'.format(dice))
  
         
        if not config.debug:
            outdir = config.dir_seg + \
                '/{}/{}'.format(self.model_name,test)
            logger.info('saving data in: {}'.format(outdir))
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
        
            io_analyze.save(outdir + 'sol.hdr', sol.astype(np.int32))
            np.save(outdir + 'y.npy', y)
        
            np.savetxt(
                outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %.8f')
        
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
            

            
            
if __name__=='__main__':
    ''' start script '''
    #sample_list = ['01/']
    sample_list = ['02/']
    #sample_list = config.vols
   
    # combine entropy / intensity
    segmenter = SegmentationBatch()
    segmenter.process_all_samples(sample_list)
    

    
    
    
    
    
