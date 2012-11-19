import numpy as np
from scipy import ndimage
import rwsegment
reload(rwsegment)
from rwsegment import BaseAnchorAPI

import  utils_logging
logger = utils_logging.get_logger('rwsegment_prior_models',utils_logging.DEBUG)

'''
class PriorModel(BaseAnchorAPI):
    def __init__(self,*args, **kwargs):
        super(PriorModel,self).__init__(*args, **kwargs)
        self.init_model()
    def init_model(self):
        pass
    def get_anchor_and_weights(self, D, indices):
        nlabel = len(self.labelset)
        N = np.maximum(np.max(self.imask), np.max(indices))+ 1
        data = 1./nlabel * np.ones((nlabel,N))
        data[:,self.imask] = self.anchor['data']
        data = data[:,indices]
        weights = self.anchor_weight * np.ones((nlabel,len(indices)))
        return data, weights
'''

class Constant(BaseAnchorAPI):
    def get_anchor_and_weights(self, D, indices):
        data, weights = super(Constant, self).get_anchor_and_weights(1, indices)
        return data, weights # *1
 
class Uniform(BaseAnchorAPI):
    pass
    
class Entropy(BaseAnchorAPI):
    def __init__(self,*args, **kwargs):
        super(Entropy,self).__init__(*args, **kwargs)
        nlabel = len(self.anchor['data'])
        prior = np.asarray(self.anchor['data'])
        entropy = -np.sum(np.log(prior + 1e-10)*prior,axis=0)
        entropy[entropy<0] = 0
        self.entropy = \
            np.tile((np.log(nlabel) - entropy) / np.log(nlabel),(nlabel,1))
    def get_anchor_and_weights(self, D, indices):
        data, weights = super(Entropy, self).get_anchor_and_weights(D,indices)
        inds1 = np.in1d(indices, self.imask)
        inds2 = np.in1d(self.imask, indices)
        weights[:,inds1] *= self.entropy[:,inds2]
        return data, weights * D

class Entropy_no_D(Entropy):
    def get_anchor_and_weights(self, D, indices):
        return super(Entropy_no_D, self).get_anchor_and_weights(1, indices)
    
class Variance(BaseAnchorAPI):
    def get_anchor_and_weights(self, D, indices):
        data, weights = super(Variance, self).get_anchor_and_weights(D,indices)
        inds1 = np.in1d(indices, self.imask)
        inds2 = np.in1d(self.imask, indices)
        wvar = 1./ (1 + np.asarray(self.anchor['variance'])[:,inds2])
        wvar /= np.max(wvar)
        weights[:,inds1] *= wvar
        return data, weights * D
   
class Variance_no_D(Variance):
    def get_anchor_and_weights(self, D, indices):
        return super(Variance_no_D, self).get_anchor_and_weights(1, indices)
 
class Variance_no_D_Cmap(BaseAnchorAPI):
    def __init__(self, *args,**kwargs):
        im = kwargs.pop('image')
        super(Variance_no_D_Cmap,self).__init__(*args, **kwargs)
        
        fim = ndimage.gaussian_gradient_magnitude(im,2)
        fim = fim/np.std(fim)
        alpha = 1e0
        self.cmap = np.exp(-fim.flat[self.anchor['imask']]*alpha) + 1e-10

    def get_anchor_and_weights(self, D, indices):
        data, weights = super(Variance_no_D_Cmap, self).get_anchor_and_weights(1,indices)
        inds1 = np.in1d(indices, self.imask)
        inds2 = np.in1d(self.imask, indices)
        wvar = self.cmap[:,inds2] / (1 + np.asarray(self.anchor['variance'])[:,inds2])
        wvar /= np.max(wvar)
        weights[:,inds1] *= wvar
        return data, weights * D

   
class Intensity(BaseAnchorAPI):
    def __init__(self, *args,**kwargs):
        self.image = kwargs.pop('image')
        super(Intensity,self).__init__(*args, **kwargs)
        
        ## classify image
        nlabel = len(self.labelset)
        avg,var = self.anchor['intensity']
        diff = self.image.flat - np.c_[avg]
        norm = 1./np.sqrt(2*np.pi*var)
        a = np.c_[norm] * np.exp( - 0.5*diff**2 * np.c_[1./var] )
        A = np.clip(np.sum(a, axis=0),1e-10,np.inf)
        self.data = (1./A)*a
        self.intensity = np.tile(A, (nlabel,1))
        
    def get_anchor_and_weights(self, D, indices):
        nlabel = len(self.labelset)
        data = 1./nlabel * np.ones((nlabel,len(indices)))
        weights = self.anchor_weight * np.ones((nlabel,len(indices)))
        inds1 = np.in1d(indices, self.imask)
        inds2 = np.in1d(self.imask, indices)
        data[:,inds1] = self.data[:,inds2]
        weights[:,inds1] *= self.intensity[:,inds2]
        return data, weights

        