import sys
import os
import numpy as np


from scipy import sparse
from scipy.sparse import linalg as splinalg


## compute prior 
class PriorGenerator:
    ''' generate prior
    '''
    
    def __init__(self, labelset):
        self.x0     = 0
        #square(x) is useless on binary x...
        # self.x02    = 0
        self.ntrain = 0
        self.mask   = 0
        
        ## intensity prior
        self.im_ntrain = None
        
        self.labelset = np.asarray(labelset)
        
        
    def add_training_data(self, atlas, image=None, **kwargs):
        a = np.asarray(atlas, dtype=int)
        
        ## set unwanted labels to background label (labelset[0])
        bg = self.labelset[0]
        a.flat[~np.in1d(a, self.labelset)] = bg
        
        ## compute background mask
        self.mask = self.mask | (a!=bg)
        
        ## binary assignment matrix
        bin = a.ravel()==np.c_[self.labelset]
        
        ## compute average
        x = bin.astype(float)
        self.x0     = self.x0 + x
        self.ntrain += 1
        
        ## if im is provided, compute average and std of intensity
        if image is not None:
            nim = image
            if self.mask is not None:
                nim = nim[self.mask]
            nim = nim/np.std(nim)
            if self.im_ntrain is None:
                self.im_avg    = np.zeros(len(self.labelset))
                self.im_avg2   = np.zeros(len(self.labelset))
                self.im_ntrain = np.zeros(len(self.labelset), dtype=int)
                
            for label in range(len(self.labelset)):
                inds = np.where(bin[label])[0]
                self.im_avg[label]    += np.sum(nim.flat[inds])
                self.im_avg2[label]   += np.sum(nim.flat[inds]**2)
                self.im_ntrain[label] += len(inds)
                
        
    def get_mask(self):
        return self.mask
        
    def get_prior(self,mask):
        nlabel = len(self.labelset)
        imask = np.where(mask.ravel())[0]
        
        ## average probability
        mean = self.x0[:,imask] / float(self.ntrain)
        
        ## variance
        # x is binary, thus we have avg(x) = avg(x**2)
        var = mean  - mean**2

        ## prior dict
        prior = {
            'labelset': self.labelset,
            'imask': imask, 
            'data': mean, 
            'variance': var,
            }
        
        ## if intensity prior
        if self.im_ntrain is not None:
            im_avg = self.im_avg / self.im_ntrain.astype(float)
            im_var = self.im_avg2 / self.im_ntrain.astype(float) - im_avg**2
            
            ## add to prior dict
            prior['intensity']= (im_avg, im_var)
        
        return prior
        
