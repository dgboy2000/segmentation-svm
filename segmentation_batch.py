import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment
from rwsegment import rwsegment_prior_models as prior_models
reload(rwsegment)
reload(prior_models)


import segmentation_utils 
reload(segmentation_utils)

from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
    
import config
reload(config)


class SegmentationBatch(object):
    
    def __init__(self,wprior=1.,model_type='constant'):
        ''' 
            model_type in 'constant', 'entropy', 'variance', 'cmap'
        '''
        
        self.labelset  = np.asarray(config.labelset)
        self.model_type = model_type
        self.force_recompute_prior = False
        
        self.params  = {
            'wprior'    : wprior, # prior weight
            'beta'      : 50,     # contrast parameter
            'return_arguments':['image'],
            
            # optimization parameter
            'per_label': False,
            'optim_solver':'unconstrained',
            'rtol'      : 1e-6, 
            'maxiter'   : 1e3,
            }
            
        if self.model_type=='constant': 
            self.prior_function = prior_models.constant
        elif self.model_type=='uniform': 
            self.prior_function = prior_models.uniform
        elif self.model_type=='entropy': 
            self.prior_function = prior_models.entropy
        elif self.model_type=='entropy_no_D': 
            self.prior_function = prior_models.entropy_no_D
        elif self.model_type=='variance': 
            self.prior_function = prior_models.variance
        elif self.model_type=='variance_no_D': 
            self.prior_function = prior_models.variance_no_D
        elif self.model_type=='confidence_map': 
            self.prior_function = prior_models.confidence_map
        elif self.model_type=='confidence_map_no_D': 
            self.prior_function = prior_models.confidence_map_no_D
        else:
            raise Exception('Did not recognize prior model type: {}'\
                .format(self.model_type))
        logger.info('using prior model: {}, with weight={}'\
            .format(self.model_type, wprior))
    
    def process_sample(self,test):
        outdir = config.dir_work + \
            'segmentation/{}/{}'.format(self.model_type,test)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        ## get prior
        prior, mask = load_or_compute_prior_and_mask(
            test,force_recompute=self.force_recompute_prior)
        
        ## segment image
        file_name = config.dir_reg + test + 'gray.hdr'        
        print 'segmenting data: {}'.format(file_name)
        im      = io_analyze.load(file_name)
        seeds   = (-1)*mask
           
        ## normalize image
        im = im/np.std(im)
            
        prior_function = self.prior_function(
            prior=prior,
            image=im,
            )
            
        ## start segmenting
        sol = rwsegment.segment(
            im, 
            prior, 
            # prior_weights=prior_weights,
            prior_function=prior_function,
            seeds=seeds, 
            labelset=self.labelset, 
            **self.params
            )
            
        io_analyze.save(outdir + 'sol.hdr', sol.astype(np.int32))    
        
        ## compute Dice coefficient
        file_gt = config.dir_reg + test + 'seg.hdr'
        seg     = io_analyze.load(file_gt)
        dice = compute_dice_coef(sol, seg,labelset=self.labelset)
        np.savetxt(
            outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %f')
        
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
            
from rwsegment import utils_logging
logger = utils_logging.get_logger('segmentation_batch',utils_logging.INFO)
            
            
if __name__=='__main__':
    ''' start script '''
    # segmenter = SegmentationBatch(wprior=1e-2,model_type='constant')
    segmenter = SegmentationBatch(wprior=0.5,model_type='uniform')
    # segmenter = SegmentationBatch(wprior=0.5, model_type='entropy')
    # segmenter = SegmentationBatch(wprior=1e-2, model_type='entropy_no_D')
    # segmenter = SegmentationBatch(model_type='variance')
    
    sample_list = ['01/']
    # sample_list = config.vols
    segmenter.process_all_samples(sample_list)
    
    