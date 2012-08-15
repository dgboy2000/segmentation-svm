import numpy as np

import  utils_logging
logger = utils_logging.get_logger('rwsegment_prior_models',utils_logging.DEBUG)

class PriorModel(object):
    def __init__(self, **kwargs):
        for attr in kwargs:
            setattr(self, attr,kwargs[attr])
    def __call__(self,D):
        return np.zeros(D.size)

class constant(PriorModel):
    def __call__(self, D,**kwargs):
        mean = np.asarray(self.prior['mean'])
        shape = mean.shape
        weights = np.ones(shape)
        return weights
    
class uniform(PriorModel):
    def __call__(self, D,**kwargs):
        mean = np.asarray(self.prior['mean'])
        shape = mean.shape
        weights = D * np.ones(shape)
        return weights
    
class entropy(PriorModel):
    def entropy(self):
        nlabel = len(self.prior['mean'])
        mean = np.asarray(self.prior['mean'])
        entropy = -np.sum(np.log(mean + 1e-10)*mean,axis=0)
        entropy[entropy<0] = 0
        return entropy
    def __call__(self, D,**kwargs):
        entropy = self.entropy()
        weights = np.tile(D * (np.log(nlabel) - entropy)/np.log(nlabel),(nlabel,1))
        return weights

class entropy_no_D(entropy):
    def __call__(self, D,**kwargs):
        _D = np.ones(D.size)
        return super(_D,**kwargs)
    
class variance(entropy):
    def __call__(self, D,**kwargs):
        var = np.asarray(self.prior['var'])
        weights = D * 1/(1.0 + var)
        return weights
    
class variance_no_D(variance):
    def __call__(self, D,**kwargs):
        _D = np.ones(D.size)
        return super(_D,**kwargs)
    
class confidence_map(PriorModel):
    def __call__(self, D, **kwargs):
        image = self.image
        #...
        #cmap = image
        nlabel = len(self.prior['mean'])
        indices = np.asarray(self.prior['indices'])
        weights = np.tile(D*cmap.flat[indices],(nlabel,1))
        return weights
        
class confidence_map_no_D(confidence_map):
    def __call__(self, D,**kwargs):
        _D = np.ones(D.size)
        return super(_D,**kwargs)
    
    