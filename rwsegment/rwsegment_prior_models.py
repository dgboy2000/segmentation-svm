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
        anchor, weights = super(Constant, self).get_anchor_and_weights(i, D_,**kwargs)
        return anchor, weights # *1
 
class Constant_Cmap(Constant):
    def make_cmap(self, im):
        fim = ndimage.gaussian_gradient_magnitude(im,2)
        fim = fim/np.std(fim)
        alpha = 1e0
        cmap = np.exp(-fim.ravel()*alpha) + 1e-10
        return cmap

    def get_anchor_and_weights(self, i, D, image=None, **kwargs):
        anchor, weights = super(Constant_Cmap, self).get_anchor_and_weights(i, D, **kwargs)
        return anchor, weights* self.make_cmap(image)[i]


class Uniform(BaseAnchorAPI):
    pass
    
class Entropy(BaseAnchorAPI):
    def __init__(self, *args, **kwargs):
        super(Entropy,self).__init__(*args, **kwargs)
        nlabel = len(self.anchor)
        entropy = -np.sum(np.log(self.anchor + 1e-10)*self.anchor,axis=0)
        entropy[entropy<0] = 0
        #entropy = (np.log(nlabel) - entropy) / np.log(nlabel)
        entropy = (np.max(entropy) - entropy) / np.max(entropy) # temp
        self.weights = entropy

class Entropy_no_D(Entropy):
    def get_anchor_and_weights(self, i, D, **kwargs):
        D_ = np.ones(np.asarray(D).shape)
        anchor, weights = super(Entropy, self).get_anchor_and_weights(i, D_, **kwargs)
        return anchor, weights
    
class Variance(BaseAnchorAPI):
    def __init__(self, ianchor, anchor, variance):
        #1/0
        weights = np.min(1. / (1. + variance), axis=0)
        super(Variance,self).__init__(ianchor, anchor, weights=weights)
  
class Variance_no_D(Variance):
    def get_anchor_and_weights(self, i, D, **kwargs):
        D_ = np.ones(np.asarray(D).shape)
        return super(Variance_no_D, self).get_anchor_and_weights(i, D_, **kwargs)
 
class Variance_no_D_Cmap(Variance):
    def make_cmap(self, im):
        fim = ndimage.gaussian_gradient_magnitude(im,2)
        fim = fim/np.std(fim)
        alpha = 1e0
        cmap = np.exp(-fim.ravel()*alpha) + 1e-10
        return cmap

    def get_anchor_and_weights(self, i, D, image=None, **kwargs):
        D_ = np.ones(np.asarray(D).shape)
        anchor, weights = super(Variance_no_D_Cmap, self).get_anchor_and_weights(i, D_, **kwargs)
        if image is not None:
            weights *= self.make_cmap(image)[i]
        return anchor, weights

        
class Spatial(Constant):
    def __init__(self, ianchor, anchor, sweights=1.):
        nlabel = len(anchor)
        sw = np.ones(nlabel)*sweights
        #smap = sum([ (1-anchor[s])*(1-sw[s]) + anchor[s]*sw[s] for s in range(nlabel)]) / (nlabel - np.sum(sw) + 1)
        #smap = np.clip(np.min((1-anchor) + np.c_[sw], axis=0),0,1)
        #smap = np.clip((1-anchor[0]) + sw[0], 0, 1)
        #weights = np.tile(smap, (nlabel,1))
        #anchor2 = anchor * smap + (1 - smap)/float(nlabel)
        #anchor2 = anchor * np.c_[sweights]
        #anchor2 = anchor2/np.sum(anchor2, axis=0)
        #import ipdb; ipdb.set_trace()
        super(Spatial,self).__init__(ianchor, anchor)
        #super(Spatial,self).__init__(ianchor, anchor2)
    
   
class Intensity(object):
    def __init__(self, im_avg, im_var):
        self.im_avg = im_avg
        self.im_var = im_var
       
    def get_anchor_and_weights(self, i, D, **kwargs):
        image = kwargs.pop('image')
        nlabel = len(self.im_avg)
        ## classify image
        diff = image.flat[i] - np.c_[self.im_avg]
        norm = 1./np.sqrt(2*np.pi*self.im_var)
        a = np.c_[norm] * np.exp( - diff**2 * np.c_[1./self.im_var] )
        A = np.sum(a, axis=0)
        anchor = (1./A)*a
        #weights = np.tile(A, (nlabel,1))
        weights = A
        return anchor, weights

        

        
