import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment
from rwsegment.rwsegment import  BaseAnchorAPI
reload(rwsegment)

import segmentation_utils 
reload(segmentation_utils)
from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
    
import config
reload(config)

from rwsegment import utils_logging
logger = utils_logging.get_logger('autoseeds',utils_logging.INFO)

class Autoseeds(object):
    
    def __init__(self):
        
        self.labelset  = np.asarray(config.labelset)
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
        
    def process_sample(self,test):

        ## get prior
        prior, mask = load_or_compute_prior_and_mask(
            test,force_recompute=self.force_recompute_prior)
        
        ## load image
        file_name = config.dir_reg + test + 'gray.hdr'        
        logger.info('segmenting data: {}'.format(file_name))
        im      = io_analyze.load(file_name)
        file_gt = config.dir_reg + test + 'seg.hdr'
        seg     = io_analyze.load(file_gt)
        seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
        
           
        ## normalize image
        nim = im/np.std(im)
        
        ## sample unlabeled seeds
        from rwsegment import boundary_utils
        seed_inds = boundary_utils.sample_points(nim, step=[5,10,10], mask=mask)

        ## set unary potentials from prior: array of unary costs
        nlabel = len(self.labelset)
        prob = np.zeros((nim.size, nlabel))
        prob[mask.ravel(),:] = prior['data']
        unary = - np.log(prob[tuple(seed_inds.T),:] + 1e-10)
        
        ## set binary potentials: list of [(ind neightbor,  binary costs), etc.]
        binary = boundary_utils.boundary_probability(nim, seed_inds)
       
        ## solve MRF

  
        ## start segmenting
        sol,y = rwsegment.segment(
            nim, 
            seeds=seeds, 
            labelset=self.labelset, 
            weight_function=self.weight_function,
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
        
            np.savetxt(
                outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %.8f')
        
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
            

            
            
if __name__=='__main__':
    ''' start script '''
    sample_list = ['M44/']
    #sample_list = config.vols
   
    # Autoseeds
    segmenter = Autoseeds()
    segmenter.process_all_samples(sample_list)
    

    
    
     
