import numpy as np
from rwsegment import rwmean_svm
reload(rwmean_svm)
        
## loss function
class Loss(object):
    def __init__(self, nlabel):
        self.nlabel = nlabel
    def __call__(self,z,y_):
        ''' 1 - (z.y_)/nnode '''
        if np.sum(y_<-1e-8) > 0:
            logger.error('negative values in y_')
    
        nnode = z.size/float(self.nlabel)
        return 1.0 - 1.0/nnode * np.dot(z,y_)

## psi
class Psi(object):
    def __init__(self, 
            prior, labelset, weight_functions, rwparams, **kwargs):
        self.prior = prior
        self.labelset = labelset
        self.weight_functions = weight_functions
        self.seeds = kwargs.pop('seeds', [])
        self.rwparams = rwparams
        
    def __call__(self, x,y):
        ''' - sum(a){Ea(x,y)} '''
        
        ## energy value for each weighting function
        v = []
        for wf in self.weight_functions.values():
            
            v.append(
                rwmean_svm.energy_RW(
                    x,self.labelset,y,
                    weight_function=wf, 
                    seeds=self.seeds,
                    **self.rwparams
                ))
                
        ## last coef is prior
        v.append(
            rwmean_svm.energy_mean_prior(
                self.prior,y,
                seeds=self.seeds,
                **self.rwparams
            ))
            
        ## psi[a] = minus energy[a]
        return v


## most violated constraint
class Most_violated_constraint(object):
    def __init__(
            self, prior, 
            weight_functions, 
            labelset, 
            rwparams,
            **kwargs):
        self.prior = prior
        self.weight_functions = weight_functions
        self.labelset = labelset
        self.rwparams = rwparams
        self.seeds = kwargs.pop('seeds', [])
        
    def __call__(self,w,x,z):
        ''' y_ = arg min <w|-psi(x,y_)> - loss(y,y_) '''
        
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
        labelset = np.asarray(self.labelset)
        linloss = 1./nnode*z
        
        ## best y_ most different from y
        y_ = rwmean_svm.segment_mean_prior(
            x, 
            self.prior, 
            seeds=self.seeds,
            weight_function=lambda im: wwf(im, w),
            add_linear_term=linloss,
            lmbda=w[-1],
            **self.rwparams
            )
            
        return y_
        
        
        
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