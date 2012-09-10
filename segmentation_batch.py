import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment
from rwsegment import weight_functions as wflib
from rwsegment import rwsegment_prior_models as prior_models
from rwsegment import loss_functions
from rwsegment.rwsegment import  BaseAnchorAPI
from svm_rw_api import MetaAnchor
reload(rwsegment)
reload(loss_functions)
reload(prior_models)


import segmentation_utils 
reload(segmentation_utils)

from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
    
import config
reload(config)

from rwsegment import utils_logging
logger = utils_logging.get_logger('segmentation_batch',utils_logging.INFO)

class SegmentationBatch(object):
    
    #def __init__(self, anchor_weight=1., model_type='constant'):
    def __init__(self, prior_weights=[1.,0,0], name='constant1'):
        
        self.labelset  = np.asarray(config.labelset)
        #self.model_type = model_type
        self.model_name = name
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
        
        laplacian_type = 'std_b50'
        #laplacian_type = 'inv_b100o1'
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
       
        self.prior_models = [
            prior_models.Constant,
            prior_models.Entropy_no_D,
            prior_models.Intensity,
            ]
        self.prior_weights = prior_weights

        #self.anchor_weight = anchor_weight
        #if self.model_type=='constant': 
        #    self.Model = prior_models.Constant
        #elif self.model_type=='uniform': 
        #    self.Model = prior_models.Uniform
        #elif self.model_type=='entropy': 
        #    self.Model = prior_models.Entropy
        #elif self.model_type=='entropy_no_D': 
        #    self.Model = prior_models.Entropy_no_D
        #elif self.model_type=='variance': 
        #    self.Model = prior_models.Variance
        #elif self.model_type=='variance_no_D': 
        #    self.Model = prior_models.Variance_no_D
        #elif self.model_type=='confidence_map': 
        #    self.Model = prior_models.Confidence_map
        #elif self.model_type=='confidence_map_no_D': 
        #    self.Model = prior_models.Confidence_map_no_D
        #elif self.model_type=='intensity': 
        #    self.Model = prior_models.Intensity
        #elif self.model_type=='combined': 
        #    self.Model = prior_models.CombinedConstantIntensity
        #else:
        #    raise Exception('Did not recognize prior model type: {}'\
        #        .format(self.model_type))

        logger.info('Model name = {}, using prior weights={}'\
            .format(self.model_name, self.prior_weights))
    
    def process_sample(self,test):

        ## get prior
        prior, mask = load_or_compute_prior_and_mask(
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
            
        ## init anchor_api
        anchor_api = MetaAnchor(
            prior=prior,
            prior_models=self.prior_models,
            prior_weights=self.prior_weights,
            image=im,
            )
         #anchor_api = self.Model(
         #   prior,
         #   anchor_weight=self.anchor_weight,
         #   image=im,
         #   )
            
        ## start segmenting
        # import ipdb; ipdb.set_trace()
        sol,y = rwsegment.segment(
            nim, 
            anchor_api,
            seeds=seeds, 
            labelset=self.labelset, 
            weight_function=self.weight_function,
            **self.params
            )

        ## compute losses
        z = seg.ravel()==np.c_[self.labelset]
        flatmask = mask.ravel()*np.ones((len(self.labelset),1))
        
        ## loss 0 : 1 - Dice(y,z)
        loss0 = loss_functions.ideal_loss(z,y,mask=flatmask)
        logger.info('loss0 (Dice) = {}'.format(loss0))
        
        ## loss2: squared difference with ztilde
        loss1 = loss_functions.anchor_loss(z,y,mask=flatmask)
        logger.info('loss1 (anchor) = {}'.format(loss1))
        
        ## loss3: laplacian loss
        loss2 = loss_functions.laplacian_loss(z,y,mask=flatmask)
        logger.info('loss2 (laplacian) = {}'.format(loss2))
        
        ## compute Dice coefficient per label
        dice    = compute_dice_coef(sol, seg,labelset=self.labelset)
        logger.info('Dice: {}'.format(dice))
        
        if not config.debug:
            #outdir = config.dir_seg + \
            #    '/{}/{}'.format(self.model_type,test)
            outdir = config.dir_seg + \
                '/{}/{}'.format(self.model_name,test)
            logger.info('saving data in: {}'.format(outdir))
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
        
            io_analyze.save(outdir + 'sol.hdr', sol.astype(np.int32))
            np.save(outdir + 'y.npy', y)
        
            f = open(outdir + 'losses.txt', 'w')
            f.write('ideal_loss\t{}\n'.format(loss0))
            f.write('anchor_loss\t{}\n'.format(loss1))
            f.write('laplacian_loss\t{}\n'.format(loss2))
            f.close()
            
            np.savetxt(
                outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %.8f')
        
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
            

            
            
if __name__=='__main__':
    ''' start script '''
    #sample_list = ['01/']
    #sample_list = ['02/']
    sample_list = config.vols
   
    # constant prior
    #segmenter = SegmentationBatch(anchor_weight=1e-2 ,model_type='constant')
    segmenter = SegmentationBatch(prior_weights=[1e-2, 0, 0], name='constant1e-2')
    segmenter.process_all_samples(sample_list)
    
    # entropy prior
    segmenter = SegmentationBatch(prior_weights=[0, 1e-2, 0], name='entropy1e-2')
    segmenter.process_all_samples(sample_list)
    
    # entropy prior
    segmenter = SegmentationBatch(prior_weights=[0, 1e-1, 0], name='entropy1e-1')
    segmenter.process_all_samples(sample_list)
    

    # intensity prior
    #segmenter = SegmentationBatch(anchor_weight=1.0,    model_type='intensity')
    #segmenter.process_all_samples(sample_list)
    

    # combine entropy / intensity
    segmenter = SegmentationBatch(prior_weights=[0, 1e-2, 1e-2], name='entropy1e-2_intensity1e-2')
    segmenter.process_all_samples(sample_list)
    
     # combine entropy / intensity
    segmenter = SegmentationBatch(prior_weights=[0, 1e-3, 1e-2], name='entropy1e-3_intensity1e-2')
    segmenter.process_all_samples(sample_list)
    

    
    
    
    
    
