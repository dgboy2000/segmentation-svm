import numpy as np

import rwsegment
reload(rwsegment)
from rwsegment import BaseAnchorAPI

import  utils_logging
logger = utils_logging.get_logger('rwsegment_prior_models',utils_logging.DEBUG)

class PriorModel(BaseAnchorAPI):
    def __init__(self,*args, **kwargs):
        super(PriorModel,self).__init__(*args, **kwargs)
        self.init_model()
    def init_model(self):
        pass
    def get_anchor_and_weights(self, D):
        nlabel = len(self.labelset)
        return self.anchor, np.zeros((nlabel,D.size))

class Constant(PriorModel):
    def get_anchor_and_weights(self, D):
        nlabel = len(self.labelset)
        weights = self.anchor_weight * np.ones((nlabel,D.size))
        return self.anchor, weights
    
class Uniform(PriorModel):
    def get_anchor_and_weights(self, D):
        nlabel = len(self.labelset)
        weights = self.anchor_weight * D * np.ones((nlabel,D.size))
        return self.anchor, weights
    
class Entropy(PriorModel):
    def init_model(self):
        nlabel = len(self.anchor['data'])
        prior = np.asarray(self.prior['data'])
        entropy = -np.sum(np.log(prior + 1e-10)*prior,axis=0)
        entropy[entropy<0] = 0
        self.weights = \
            np.tile((np.log(nlabel) - entropy) / np.log(nlabel),(nlabel,1))
    def get_anchor_and_weights(self, D):
        weights = self.anchor_weight * D * self.weights
        return self.anchor, weights

class Entropy_no_D(Entropy):
    def get_anchor_and_weights(self, D):
        weights = self.anchor_weight * self.weights
        return self.anchor, weights
    
class Variance(PriorModel):
    def get_anchor_and_weights(self, D):
        weights = self.anchor_weight * D * np.asarray(self.anchor['variance'])
        return self.anchor, weights
    
class Variance_no_D(PriorModel):
    def get_anchor_and_weights(self, D):
        weights = self.anchor_weight * np.asarray(self.anchor['variance'])
        return self.anchor, weights
    
class Intensity(PriorModel):
    def __init__(self, *args,**kwargs):
        self.image = kwargs.pop('image')
        super(PriorModel,self).__init__(*args, **kwargs)
        self.init_model()
        
    def init_model(self):
        ## classify image
        nlabel = len(self.labelset)
        avg,var = self.anchor['intensity']
        diff = self.image.flat[self.imask] - np.c_[avg]
        norm = 1./np.sqrt(2*np.pi*var)
        a = np.c_[norm] * np.exp( - diff**2 * np.c_[1./var] )
        A = np.sum(a, axis=0)
        self.prior   = {
            'imask': self.anchor['imask'],
            'data': (1./A)*a,
            }
        self.weights = np.tile(A, (nlabel,1))
        
    def get_anchor_and_weights(self, D):
        return self.prior, self.anchor_weight * self.weights

        
class CombinedConstantIntensity(Intensity):
    def init_model(self):
        super(CombinedConstantIntensity,self).init_model()
        nlabel, nnode = len(self.anchor['data']), len(self.anchor['data'][0])
        
        # constant prior
        cprior = self.anchor['data']
        cweights = np.ones((nlabel,nnode))
        
        # intensity prior
        iprior = self.prior['data']
        iweights = self.weights / np.mean(self.weights)
        
        # combined weights
        weights = cweights + iweights
        prior = (cweights*cprior + iweights*iprior) / weights
        
        self.prior = {
            'imask': self.anchor['imask'],
            'data': prior,
            }
        self.weigths = weights
        
        
        
class Confidence_map(PriorModel):
    def __init__(self,image, *args, **kwargs):
        self.image = kwargs.pop('image')
        super(PriorModel,self).__init__(*args, **kwargs)

        #...
        #cmap = image
        nlabel = len(self.prior['mean'])
        indices = np.asarray(self.prior['indices'])
        weights = np.tile(D*cmap.flat[indices],(nlabel,1))
        self.weights = weights
    def get_anchor_and_weights(self, D):
        weights = self.anchor_weight * D * self.weights
        return self.anchor, weights
        
class Confidence_map_no_D(Confidence_map):
    def get_anchor_and_weights(self, D):
        weights = self.anchor_weight * self.weights
        return self.anchor, weights
