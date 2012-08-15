import numpy as np
from rwsegment import rwmean_svm
reload(rwmean_svm)
        

class SVMRWMeanAPI(object):
    def __init__(
            self, 
            prior, 
            weight_functions, 
            labelset, 
            rwparams, 
            **kwargs):
    
        self.prior = prior
        self.labelset = labelset
        self.nlabel = len(labelset)
        self.weight_functions = weight_functions
        self.seeds = kwargs.pop('seeds', [])
        self.rwparams = rwparams
    
## loss function
# class Loss(object):
    # def __init__(self, nlabel):
        # self.nlabel = nlabel
    # def __call__(self,z,y_):
    def compute_loss(self,z,y_):
        ''' 1 - (z.y_)/nnode '''
        if np.sum(y_<-1e-8) > 0:
            logger.error('negative values in y_')
    
        nnode = z.size/float(self.nlabel)
        return 1.0 - 1.0/nnode * np.dot(z,y_)

## psi
# class Psi(object):
    # def __init__(self, 
            # prior, labelset, weight_functions, rwparams, **kwargs):
        # self.prior = prior
        # self.labelset = labelset
        # self.weight_functions = weight_functions
        # self.seeds = kwargs.pop('seeds', [])
        # self.rwparams = rwparams
        
    # def __call__(self, x,y):
    def compute_psi(self, x,y):
        ''' - sum(a){Ea(x,y)} '''
        ## normalizing by the approximate mask size
        nnode = x.size/10.0
        
        ## energy value for each weighting function
        v = []
        for wf in self.weight_functions.values():
            
            v.append( 
                rwmean_svm.energy_RW(
                    x,self.labelset,y,
                    weight_function=wf, 
                    seeds=self.seeds,
                    **self.rwparams
                )/float(nnode))
                
        ## last coef is prior
        v.append(
            rwmean_svm.energy_mean_prior(
                self.prior,y,
                seeds=self.seeds,
                **self.rwparams
            )/float(nnode))
            
        ## psi[a] = minus energy[a]
        return v


## most violated constraint
# class Most_violated_constraint(object):
    # def __init__(
            # self, prior, 
            # weight_functions, 
            # labelset, 
            # rwparams,
            # **kwargs):
        # self.prior = prior
        # self.weight_functions = weight_functions
        # self.labelset = labelset
        # self.rwparams = rwparams
        # self.seeds = kwargs.pop('seeds', [])
        
        
    def full_lai(self, w,x,z):
        ''' full Loss Augmented Inference
         y_ = arg min <w|-psi(x,y_)> - loss(y,y_) '''
        
        ## combine all weight functions
        def wwf(im,_w):    
            ''' meta weight function'''
            data = 0
            for iwf,wf in enumerate(self.weight_functions.values()):
                ij,_data = wf(im)
                data += _w[iwf]*_data
            return ij, data
        
        ## loss as a linear term in the function
        nnode = x.size
        linloss = 1./nnode*z
        
        ## best y_ most different from y
        y_ = rwmean_svm.segment_mean_prior(
            x, 
            self.prior, 
            seeds=self.seeds,
            weight_function=lambda im: wwf(im, w),
            add_linear_term=linloss,
            # prior_weights=prior_weights,
            lmbda=w[-1],
            **self.rwparams
            )
            
        return y_
        
    def worst_loss_inference(self, x,z):
        nlabel = len(self.labelset)
        y_ = (1-z)/float(nlabel-1)
        return y_
        
    def best_segmentation_inference(self, w,x):
    
        ## combine all weight functions
        def wwf(im,_w):    
            ''' meta weight function'''
            data = 0
            for iwf,wf in enumerate(self.weight_functions.values()):
                ij,_data = wf(im)
                data += _w[iwf]*_data
            return ij, data
            
        ## best y_ most different from y
        y_ = rwmean_svm.segment_mean_prior(
            x, 
            self.prior, 
            seeds=self.seeds,
            weight_function=lambda im: wwf(im, w),
            lmbda=w[-1],
            **self.rwparams
            )
        return y_
    
    # def __call__(self,w,x,z,exact=True):
    def compute_mvc(self,w,x,z,exact=True):
        if exact:
            return self.full_lai(w,x,z)
        else:
            y_loss = self.worst_loss_inference(x,z)
            y_seg  = self.best_segmentation_inference(w,x)
            
            score_loss = \
                np.dot(w,self.compute_psi(x,y_loss)) - \
                self.compute_loss(z,y_loss)
                
            score_seg = \
                np.dot(w,self.compute_psi(x,y_seg)) - \
                self.compute_loss(z,y_seg)
            
            if score_seg > score_loss:
                return y_loss
            else:
                return y_seg
        
import logging
logger = logging.getLogger('svm user functions logger')
loglevel = logging.INFO
logger.setLevel(loglevel)
# create console handler with a higher log level
if len(logger.handlers)==0:
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
else:
    logger.handlers[0].setLevel(loglevel)