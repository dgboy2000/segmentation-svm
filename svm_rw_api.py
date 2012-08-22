import numpy as np
from rwsegment import rwsegment
from rwsegment.rwsegment import BaseAnchorAPI
reload(rwsegment)

from rwsegment import utils_logging
logger = utils_logging.get_logger('svm_rw_api',utils_logging.DEBUG)

## combine all prior models
class MetaAnchor():
    def __init__(
            self, 
            prior, 
            prior_models,
            prior_weights,
            loss=None,
            loss_weight=None,
            image=None,
            ):
        self.prior = prior
        self.prior_models = prior_models
        self.prior_weights = prior_weights
        self.loss = loss
        self.loss_weight = loss_weight
        self.image = image        
        self.labelset = prior['labelset']
        
    def get_labelset(self):
        return self.labelset
        
    def get_anchor_and_weights(self,D):
        all_anchor = 0
        all_weights = 0
        
        ## prior models
        for imodel, model in enumerate(self.prior_models.values()):
            api = model(
                self.prior, 
                anchor_weight=self.prior_weights[imodel],
                image=self.image,
                )
            anchor, weights = api.get_anchor_and_weights(D)
            # anchor, weights = api.get_anchor_and_weights(1)
            all_anchor  = all_anchor  + weights * anchor['data']
            all_weights = all_weights + weights
           
        ## loss
        imask = self.prior['imask']
        if self.loss is not None:
            all_anchor  = all_anchor + \
                self.loss_weight * self.loss['data'][:,imask]
            all_weights += self.loss_weight
        
        all_anchor = all_anchor / all_weights
        labelset = self.prior['labelset']
        all_anchor_dict = {'data':all_anchor, 'imask':imask, 'labelset':labelset}
        return all_anchor_dict, all_weights 



# class LossAnchorAPI(BaseAnchorAPI):    
    # def __init__(
            # self, 
            # loss, 
            # prior, 
            # loss_weight=1.0,
            # prior_weight=1.0,
            # ):
        # BaseAnchorAPI.__init__(self, prior)
        # self.prior = prior
        
        # self.loss = loss
        # self.loss_weight = loss_weight
        # self.prior_weight = prior_weight

        
    # def get_anchor_and_weights(self,D): 
        ## constant prior for now
        # prior_weights = self.prior_weight * np.ones(D.size) * D
        
        # anchor_weights = prior_weights + self.loss_weight
        
        # prior_data = self.prior['data']
        # loss_data = self.loss['data'][:,self.imask]
        
        # anchor = {
            # 'imask':self.prior['imask'],
            # 'data':(prior_weights*prior_data + self.loss_weight*loss_data)/ \
                  # anchor_weights,
            # }
        
        # return anchor, anchor_weights
        
        
class SVMRWMeanAPI(object):
    def __init__(
            self, 
            prior, 
            weight_functions,
            labelset,
            rwparams,
            prior_models=None,  
            seeds=[],
            **kwargs):
    
        self.prior = prior
        self.labelset = labelset
        self.nlabel = len(labelset)
        
        self.weight_functions = weight_functions
        
        if prior_models is None:
            anchor_api = BaseAnchorAPI
            self.prior_models = [anchor_api]
        else:
            self.prior_models = prior_models
        
        ## indices of w
        nlaplacian = len(weight_functions)
        nprior = len(self.prior_models)
        self.indices_laplacians = np.arange(nlaplacian)
        self.indices_priors = np.arange(nlaplacian,nlaplacian + nprior)
        
        
        self.wsize = nprior + nlaplacian
        
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
        # normalize = float(nnode)/100.0
        normalize = 1.0
        
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
                
        ## energy for each prior models
        for model in self.prior_models.values():
            anchor_api = model( 
                self.prior, 
                anchor_weight=1.0,
                image=x, # intensity prior needs an image
                )
            
            v.append(
                rwsegment.energy_anchor(
                    x,y,anchor_api,
                    seeds=self.seeds,
                    weight_function=wf,
                    **self.rwparams
                )/normalize)
            
        strpsi = ' '.join('{:.3}'.format(val) for val in v)
        logger.debug('psi=[{}], normalize={:.2}'.format(strpsi,normalize))
        
        if v[0]==0:
            import ipdb; ipdb.set_trace()
        return v



    def full_lai(self, w,x,z):
        ''' full Loss Augmented Inference
         y_ = arg min <w|-psi(x,y_)> - loss(y,y_) '''
            
        ## combine all weight functions
        def meta_weight_function(im,_w):    
            ''' meta weight function'''
            data = 0
            for iwf,wf in enumerate(self.weight_functions.values()):
                ij,_data = wf(im)
                data += _w[iwf]*_data
            return ij, data
            
                
                
        ## loss function
        ztilde = (1-z) / (self.nlabel - 1.0)
        loss = {'data': ztilde}
        loss_weight = (self.nlabel - 1.0) / float(z.size)
        
        weight_function = lambda im: meta_weight_function(
            im, 
            np.asarray(w)[self.indices_laplacians],
            )

                
        anchor_api = MetaAnchor(
            prior=self.prior,
            prior_models=self.prior_models,
            prior_weights=np.asarray(w)[self.indices_priors],
            loss=loss,
            loss_weight=loss_weight,
            image=x,
            )
        
        
        ## best y_ most different from y
        y_ = rwsegment.segment(
            x, 
            anchor_api,
            seeds=self.seeds,
            weight_function=weight_function,
            return_arguments=['y'],
            **self.rwparams
            )
            
        return y_
        
    def worst_loss_inference(self, x,z):
        nlabel = len(self.labelset)
        y_ = (1-z)/float(nlabel-1)
        return y_
        
    def best_segmentation_inference(self, w,x):
    
        anchor_api = BaseAnchorAPI(
            self.prior, 
            anchor_weight=w[-1],
            )
    
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
            anchor_api,
            seeds=self.seeds,
            weight_function=lambda im: wwf(im, w),
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
