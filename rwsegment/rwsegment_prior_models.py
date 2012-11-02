import numpy as np
from scipy import ndimage
import rwsegment
reload(rwsegment)
from rwsegment import BaseAnchorAPI

import  utils_logging
logger = utils_logging.get_logger('rwsegment_prior_models',utils_logging.DEBUG)

class Constant(BaseAnchorAPI):
    def get_anchor_and_weights(self, i, D, **kwargs):
        D_ = np.ones(np.asarray(D).shape)
        anchor, weights = super(Constant, self).get_anchor_and_weights(i, D_)
        return anchor, weights # *1
 
class Uniform(BaseAnchorAPI):
    pass
    
class Entropy(BaseAnchorAPI):
    def __init__(self, *args, **args):
        super(Entropy,self).__init__(*args, **kwargs)
        nlabel = len(self.anchor)
        entropy = -np.sum(np.log(self.anchor + 1e-10)*prior,axis=0)
        entropy[entropy<0] = 0
        entropy = \
            np.tile((np.log(nlabel) - entropy) / np.log(nlabel),(nlabel,1))
        self.weights = entropy

class Entropy_no_D(Entropy):
    def get_anchor_and_weights(self, i, D, **kwargs):
        D_ = np.ones(np.asarray(D).shape)
        anchor, weights = super(Entropy, self).get_anchor_and_weights(i, D_)
    
class Variance(BaseAnchorAPI):
    def __init__(self, ianchor, anchor, variance):
        weights = 1. / (1. + variance) 
        super(Entropy,self).__init__(ianchor, anchor, weights=weights)
  
class Variance_no_D(Variance):
    def get_anchor_and_weights(self, i, D, **kwargs):
        D_ = np.ones(np.asarray(D).shape)
        return super(Variance_no_D, self).get_anchor_and_weights(i, D_)
 
class Variance_no_D_Cmap(Variance):
    def make_cmap(self, im)      
        fim = ndimage.gaussian_gradient_magnitude(im,2)
        fim = fim/np.std(fim)
        alpha = 1e0
        cmap np.exp(-fim.flat]*alpha) + 1e-10
        return cmap

    def get_anchor_and_weights(self, i, D, image=image, **kwargs):
        anchor, weights = super(Variance_no_D, self).get_anchor_and_weights(i, D_)
        weights *= self.make_cmap(image)[i]
        return anchor, weights

   
class Intensity(object):
    def __init__(self, im_avg, im_var):
        self.im_avg = im_avg
        self.im_var = im_var
       
    def get_anchor_and_weights(self, i, D, image=image, **kwargs):
        ## classify image
        diff = self.image.flat[i] - np.c_[self.im_avg]
        norm = 1./np.sqrt(2*np.pi*self.im_var)
        a = np.c_[norm] * np.exp( - diff**2 * np.c_[1./self.im_var] )
        A = np.sum(a, axis=0)
        anchor = (1./A)*a
        weights = np.tile(A, (nlabel,1))
        return anchor, weights

        
