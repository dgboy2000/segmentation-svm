import os
import numpy as np
from scipy import ndimage

from rwsegment import ioanalyze
from rwsegment import rwmean_svm
reload(rwmean_svm)

import segmentation_utils 
reload(segmentation_utils)
from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
    
import config
reload(config)


class SegmentationBatch(object):
    
    def __init__(self,model_type='constant'):
        ''' 
            model_type in 'constant', 'entropy', 'variance', 'cmap'
        '''
        
        self.labelset  = np.asarray(config.labelset)
        self.model_type = model_type
        
        self.params  = {
            'wprior'    : 1e-2, # prior weight
            'beta'      : 50, # contrast parameter
            'rtol'      : 1e-6, # optimization parameter
            'maxiter'   : 1e3,
            }
    
    def compute_prior_weights(self, prior):
        if self.model_type=='entropy':
            nlabel = len(self.labelset)
            entropy = prior['entropy']
            prior_data = (np.log(nlabel) - entropy) / np.log(nlabel)
        elif self.model_type=='constant':
            prior_data = np.ones(prior['ij'][0].size)
        else:
            raise Exception(
                'Unrecognized model type: {}'.format(self.model_type))
        
        return {'ij': prior['ij'], 'data':prior_data}
    
    def process_sample(self,test):
        outdir = config.workdir + \
            'segmentation/{}/{}'.format(self.model_type,test)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        ## get prior
        prior, mask = load_or_compute_prior_and_mask(test)
        
        ## segment image
        file_name = config.dir_reg + test + 'gray.hdr'        
        print 'segmenting data: {}'.format(file_name)
        im      = ioanalyze.load(file_name)
        seeds   = (-1)*mask
           
        ## normalize image
        im = im/np.std(im)
           
        prior_weights = self.compute_prior_weights(prior)
            
        sol = rwmean_svm.segment_mean_prior(
            im, 
            prior, 
            prior_weights=prior_weights,
            seeds=seeds, 
            labelset=self.labelset, 
            **self.params
            )
            
        ioanalyze.save(outdir + 'sol.hdr', sol.astype(np.int32))    
        
        ## compute Dice coefficient
        file_gt = config.dir_reg + test + 'seg.hdr'
        seg     = ioanalyze.load(file_gt)
        dice = compute_dice_coef(sol, seg,labelset=self.labelset)
        np.savetxt(
            outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %f')
        
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
            
if __name__=='__main__':
    ''' start script '''
    # segmenter = SegmentationBatch(model_type='constant')
    segmenter = SegmentationBatch(model_type='entropy')
    
    sample_list = ['01/']
    # sample_list = config.vols
    segmenter.process_all_samples(sample_list)
    
    