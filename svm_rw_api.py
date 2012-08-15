import numpy as np
from rwsegment import rwsegment
reload(rwsegment)
        

class SVMRWMeanAPI(object):
    def __init__(
            self, 
            prior, 
            weight_functions, 
            labelset,
            rwparams,
            seeds=[],
            **kwargs):
    
        self.prior = prior
        self.labelset = labelset
        self.nlabel = len(labelset)
        self.weight_functions = weight_functions
        self.seeds = seeds
        self.rwparams = rwparams
    

    def compute_loss(self,z,y_):
        ''' 1 - (z.y_)/nnode '''
        if np.sum(y_<-1e-8) > 0:
            logger.error('negative values in y_')
    
        nnode = z.size/float(self.nlabel)
        return 1.0 - 1.0/nnode * np.dot(z.ravel(),y_.ravel())

## psi

    def compute_psi(self, x,y):
        ''' - sum(a){Ea(x,y)} '''
        nnode = x.size
        
        ## normalizing by the approximate mask size
        normalize = float(nnode)/10.0
        
        ## energy value for each weighting function
        v = []
        for wf in self.weight_functions.values():
            
            v.append( 
                rwsegment.energy_rw(
                    x,y,
                    seeds=self.seeds,
                    weight_function=wf,
                    **self.rwparams
                )/normalize)
                
        ## last coef is prior
        v.append(
            rwsegment.energy_prior(
                x,y,self.prior,
                seeds=self.seeds,
                weight_function=wf,
                **self.rwparams
            )/normalize)
            
        ## psi[a] = -energy[a]
        return v



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
        y_ = rwsegment.segment(
            x, 
            self.prior,
            seeds=self.seeds,
            weight_function=lambda im: wwf(im, w),
            loss=linloss,
            rwprior=w[-1],
            return_arguments=['y'],
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
        y_ = rwsegment.segment(
            x, 
            self.prior,
            seeds=self.seeds,
            weight_function=lambda im: wwf(im, w),
            rwprior=w[-1],
            return_arguments=['y'],
            **self.rwparams
            )
        return y_
    
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

#-------------------------------------------------------------------------------
from rwsegment import utils_logging
logger = utils_logging.get_logger('logger_learn_svm_batch',utils_logging.DEBUG)