import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment
from rwsegment import weight_functions as wflib
from rwsegment import rwsegment_prior_models as prior_models
from rwsegment.rwsegment import  BaseAnchorAPI
reload(rwsegment)
reload(prior_models)


import segmentation_utils 
reload(segmentation_utils)

from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
    
import config
reload(config)


class SegmentationBatch(object):
    
    def __init__(self, anchor_weight=1., model_type='constant'):
        ''' 
            model_type in 'constant', 'entropy', 'variance', 'cmap'
        '''
        
        self.labelset  = np.asarray(config.labelset)
        self.model_type = model_type
        self.force_recompute_prior = False
        
        self.params  = {
            'beta'             : 50,     # contrast parameter
            'return_arguments' :['image'],
            
            # optimization parameter
            'per_label': True,
            'optim_solver':'unconstrained',
            'rtol'      : 1e-6,
            'maxiter'   : 2e3,
            }
        
        # laplacian_type = 'std_b50'
        laplacian_type = 'inv_b100o1'
        # laplacian_type = 'pdiff_r1b10'
        # laplacian_type = 'pdiff_r2b10'
        logger.info('laplacian type is: {}'.format(laplacian_type))
        
        self.weight_functions = {
            'std_b10'     : lambda im: wflib.weight_std(im, beta=10),
            'std_b50'     : lambda im: wflib.weight_std(im, beta=50),
            'std_b100'    : lambda im: wflib.weight_std(im, beta=100),
            'inv_b100o1'  : lambda im: wflib.weight_inv(im, beta=100, offset=1),
            'pdiff_r1b10': lambda im: wflib.weight_patch_diff(im, r0=1, beta=10),
            'pdiff_r2b10': lambda im: wflib.weight_patch_diff(im, r0=2, beta=10),
            'pdiff_r1b50' : lambda im: wflib.weight_patch_diff(im, r0=1, beta=50),
            }
        self.weight_function = self.weight_functions[laplacian_type]
            
        self.anchor_weight = anchor_weight
        
        if self.model_type=='constant': 
            self.Model = prior_models.Constant
        elif self.model_type=='uniform': 
            self.Model = prior_models.Uniform
        elif self.model_type=='entropy': 
            self.Model = prior_models.Entropy
        elif self.model_type=='entropy_no_D': 
            self.Model = prior_models.Entropy_no_D
        elif self.model_type=='variance': 
            self.Model = prior_models.Variance
        elif self.model_type=='variance_no_D': 
            self.Model = prior_models.Variance_no_D
        elif self.model_type=='confidence_map': 
            self.Model = prior_models.Confidence_map
        elif self.model_type=='confidence_map_no_D': 
            self.Model = prior_models.Confidence_map_no_D
        elif self.model_type=='intensity': 
            self.Model = prior_models.Intensity
        elif self.model_type=='combined': 
            self.Model = prior_models.CombinedConstantIntensity
        else:
            raise Exception('Did not recognize prior model type: {}'\
                .format(self.model_type))
        logger.info('using prior model: {}, with weight={}'\
            .format(self.model_type, anchor_weight))
    
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
        nim = im/np.std(im)
            
        ## init anchor_api
        anchor_api = self.Model(
            prior,
            anchor_weight=self.anchor_weight,
            image=im,
            )
            
        ## start segmenting
        sol = rwsegment.segment(
            nim, 
            anchor_api,
            seeds=seeds, 
            labelset=self.labelset, 
            weight_function=self.weight_function,
            **self.params
            )
            
        io_analyze.save(outdir + 'sol.hdr', sol.astype(np.int32))    
        
        ## compute Dice coefficient
        file_gt = config.dir_reg + test + 'seg.hdr'
        seg     = io_analyze.load(file_gt)
        dice    = compute_dice_coef(sol, seg,labelset=self.labelset)
        np.savetxt(
            outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %.8f')
        
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
            
from rwsegment import utils_logging
logger = utils_logging.get_logger('segmentation_batch',utils_logging.INFO)
            
            
if __name__=='__main__':
    ''' start script '''
    # segmenter = SegmentationBatch(anchor_weight=1e-2 ,model_type='constant')
    # segmenter = SegmentationBatch(anchor_weight=1e-1,    model_type='uniform')
    # segmenter = SegmentationBatch(anchor_weight=0.5,  model_type='entropy')
    segmenter = SegmentationBatch(anchor_weight=1e-2, model_type='entropy_no_D')
    # segmenter = SegmentationBatch(anchor_weight=1.0,    model_type='intensity')
    # segmenter = SegmentationBatch(anchor_weight=1.0,    model_type='combined')
    
    sample_list = ['01/']
    # sample_list = config.vols
    segmenter.process_all_samples(sample_list)
    
    
