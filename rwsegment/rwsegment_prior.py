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
        self.x02    = 0
        self.ntrain = 0
        self.mask   = 0
        
        self.labelset = np.asarray(labelset)
        
        
    def add_training_data(self, atlas):
        a = np.asarray(atlas, dtype=int)
        
        ## set unwanted labels to background label (labelset[0])
        bg = self.labelset[0]
        a.flat[~np.in1d(a, self.labelset)] = bg
        
        ## compute background mask
        self.mask = self.mask | (a!=bg)
        
        ## compute average
        x = (np.c_[a.ravel()]==self.labelset).astype(float)
        self.x0     = self.x0 + x
        self.x02    = self.x02 + x**2
        self.ntrain += 1
        
    def get_mask(self):
        return self.mask
        
    def get_prior(self,mask):
        nlabel = len(self.labelset)
        imask = np.where(mask.ravel())[0]
        
        ## average probability
        mean = self.x0[imask,:].T / np.float(self.ntrain)
        
        ## variance
        var = (self.x02[imask,:].T  - mean**2)/ np.float(self.ntrain)
        return {'imask':imask, 'mean':mean, 'var':var}
        