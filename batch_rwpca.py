import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment_pca
from rwsegment import rwsegment_prior_models as prior_models
reload(rwsegment_pca)

import segmentation_utils 
reload(segmentation_utils)

from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
    
from svm_rw_api import MetaAnchor

import config
reload(config)

from rwsegment import utils_logging
logger = utils_logging.get_logger('batch_rwpca',utils_logging.INFO)

class SegmentationBatch(object):
    
    def __init__(self, prior_weights, name='constant1'):
        
        self.labelset  = np.asarray(config.labelset)
        self.force_recompute_prior = False
        self.model_name = name
        
        self.params  = {
            'beta'             : 50,     # contrast parameter
            'return_arguments' :['image','impca'],
            
            # optimization parameter
            'per_label': True,
            'optim_solver':'unconstrained',
            'rtol'      : 1e-6,
            'maxiter'   : 2e3,
            }

        self.prior_models = [
            prior_models.Constant,
            prior_models.Entropy_no_D,
            prior_models.Intensity,
            prior_models.Variance_no_D,
            prior_models.Variance_no_D_Cmap,
            ]
        self.prior_weights = prior_weights

        logger.info('Model name = {}, using prior weights={}'\
            .format(self.model_name, self.prior_weights))
       
        
    def process_sample(self,test, fold=None):

        ## get prior
        prior, mask = load_or_compute_prior_and_mask(
            test,force_recompute=self.force_recompute_prior, pca=True, fold=fold)
        seeds   = (-1)*mask
        mask = mask.astype(bool)       
 
        ## load image
        file_name = config.dir_reg + test + 'gray.hdr'        
        logger.info('segmenting data: {}'.format(file_name))
        im      = io_analyze.load(file_name)
        file_gt = config.dir_reg + test + 'seg.hdr'
        seg     = io_analyze.load(file_gt)
        seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
        
           
        ## normalize image
        nim = im/np.std(im)
      

        ## init anchor_api
        anchor_api = MetaAnchor(
            prior=prior,
            prior_models=self.prior_models,
            prior_weights=self.prior_weights,
            image=nim,
            )
               
          
        ## start segmenting
        # import ipdb; ipdb.set_trace()
        sol,impca = rwsegment_pca.segment(
            nim, 
            anchor_api,
            seeds=seeds, 
            labelset=self.labelset, 
            **self.params
            )


        ## compute Dice coefficient per label
        dice    = compute_dice_coef(sol, seg, labelset=self.labelset)
        logger.info('Dice: {}'.format(dice))
  
        dice_pca    = compute_dice_coef(impca, seg, labelset=self.labelset)
        logger.info('Dice pca only: {}'.format(dice_pca))
         

        if not config.debug:
            if fold is not None:
                test_name = 'f{}_{}'.format(fold[0][:2],test)
            else:
                test_name = test
            outdir = config.dir_seg + \
                '/{}/{}'.format(self.model_name,test_name)

            logger.info('saving data in: {}'.format(outdir))
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
        
            io_analyze.save(outdir + 'sol.hdr', sol.astype(np.int32))
            io_analyze.save(outdir + 'solpca.hdr', impca.astype(np.int32))
        
            np.savetxt(
                outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %.8f')
            np.savetxt(
                outdir + 'dice_pca.txt', np.c_[dice.keys(),dice_pca.values()],fmt='%d %.8f')
        
    def process_all_samples(self,sample_list, fold=None):
        for test in sample_list:
            self.process_sample(test, fold=fold)
            

            
            
if __name__=='__main__':

    import sys
    if '-s' not in sys.argv: sys.exit(0)

    ''' start script '''
    for fold in config.folds:
        for w in [1e-3, 1e-2, 1e-1, 1e0, 1e1]:
            segmenter = SegmentationBatch(prior_weights=[w, 0, 0, 0,0], name='constant{}'.format(w))
            segmenter.process_all_samples(fold)
            segmenter = SegmentationBatch(prior_weights=[0, w, 0, 0,0], name='entropy{}'.format(w))
            segmenter.process_all_samples(fold)
            segmenter = SegmentationBatch(prior_weights=[1e-2, 0, w, 0,0], name='entropy1e-2_intensity{}'.format(w))
            segmenter.process_all_samples(fold)
            segmenter = SegmentationBatch(prior_weights=[0, 0, 0, w, 0], name='variance}'.format(w))
            segmenter.process_all_samples(fold)
            segmenter = SegmentationBatch(prior_weights=[0, 0, 0, 0, w], name='variance_cmap{}'.format(w))
            segmenter.process_all_samples(fold)






    ## constant prior
    #segmenter = SegmentationBatch(prior_weights=[1e-1, 0, 0, 0,0], name='constant1e-1')
    #segmenter.process_all_samples(['01/'])
    ## constant prior
    #segmenter = SegmentationBatch(prior_weights=[1e-0, 0, 0, 0,0], name='constant1e0')
    #segmenter.process_all_samples(['01/'])
    #
    ## entropy prior
    #segmenter = SegmentationBatch(prior_weights=[0, 1e-2, 0,0,0], name='entropy1e-2')
    #segmenter.process_all_samples(['01/'])
     
    
    
    
    
    
