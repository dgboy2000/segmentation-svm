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
        for imodel, model in enumerate(self.prior_models):
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
       
        if np.max(all_weights) < 1e-10:
             all_anchor = np.zeros(all_anchor.shape)
             all_weights = np.zeros(all_weights.shape)
        else: 
             all_anchor = all_anchor / all_weights
        labelset = self.prior['labelset']
        all_anchor_dict = {'data':all_anchor, 'imask':imask, 'labelset':labelset}
        return all_anchor_dict, all_weights 



## combine all weight functions
class MetaLaplacianFunction(object):
    def __init__(self,w,laplacian_functions):    
        self.w = w
        self.laplacian_functions = laplacian_functions
        
    def __call__(self,im):
        ''' meta weight function'''
        data = 0
        for iwf,wf in enumerate(self.laplacian_functions):
            ij,_data = wf(im)
            data += self.w[iwf]*_data
        return ij, data
            
        
        
class SVMRWMeanAPI(object):
    def __init__(
            self, 
            prior, 
            laplacian_functions,
            labelset,
            rwparams,
            prior_models=None,  
            seeds=[],
            **kwargs):
    
        self.prior = prior
        self.labelset = labelset
        self.nlabel = len(labelset)
        
        self.loss_type = kwargs.pop('loss_type','anchor')
        logger.info('using loss type: {}'.format(self.loss_type))
        
        self.laplacian_functions = laplacian_functions
        
        if prior_models is None:
            anchor_api = BaseAnchorAPI
            self.prior_models = [anchor_api]
        else:
            self.prior_models = prior_models
        
        ## indices of w
        nlaplacian = len(laplacian_functions)
        nprior = len(self.prior_models)
        self.indices_laplacians = np.arange(nlaplacian)
        self.indices_priors = np.arange(nlaplacian,nlaplacian + nprior)
        
        self.wsize = nprior + nlaplacian
        
        self.seeds = seeds
        self.rwparams = rwparams
        self.mask = np.asarray([seeds.ravel()<0 for i in range(self.nlabel)])

    def compute_loss(self,z,y_):
        if np.sum(y_<-1e-6) > 0:
            logger.warning('negative (<1e-6) values in y_')

        if self.loss_type == 'anchor':
            #''' 1 - (z.y_)/nnode '''
            #nnode = z.size/float(self.nlabel)
            #return 1.0 - 1.0/nnode * np.dot(z.ravel(),y_.ravel())
            '''- |y - tilde(z)|^2'''
            nlabel = self.nlabel
            mask = self.mask
            if mask is None:
                 nnode  = len(z[0])
                 mask = 1.0
            else:
                nnode = np.sum(self.mask[0])
            ztilde = (1. - np.asarray(z))/(nlabel - 1.)
            loss = 1 - np.sum(mask*(ztilde - y_)**2) * (nlabel - 1.)/float(nlabel*nnode)
        elif self.loss_type == 'laplacian':
            L = laplacian_loss(z,mask=self.mask)
            yy = np.asmatrix(np.asarray(y_).ravel()).T
            loss = 1. + float(yy.T*L*yy)
	    #if self.mask is not None:
            #    dice = 1 - np.sum(self.mask*np.abs(y_-z))\
            #        /float(np.sum(self.mask[0]))/2.0
            #else:
            #    dice = 1 - np.sum(np.abs(y_-z))/float(z.shape[1])/2.0
            #logger.debug('for dice={:.2}, loss={:.2}'.format(dice, loss))
            #import ipdb; ipdb.set_trace()
        else:
           raise Exception('wrong loss type')
           sys.exit(1)
        #logger.debug('loss={:.2}'.format( loss))
        return loss

    ## psi  
    def compute_psi(self, x,y):
        ''' - sum(a){Ea(x,y)} '''
        nnode = x.size
        
        ## normalizing by the approximate mask size
        # normalize = float(nnode)/100.0
        normalize = 1.0
        
        ## energy value for each weighting function
        v = []
        for wf in self.laplacian_functions:
            
            v.append( 
                rwsegment.energy_rw(
                    x,y,
                    seeds=self.seeds,
                    weight_function=wf,
                    **self.rwparams
                )/normalize)
                
        ## energy for each prior models
        for model in self.prior_models:
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
        # logger.debug('psi=[{}], normalize={:.2}'.format(strpsi,normalize))
        
        if v[0]==0:
            import ipdb; ipdb.set_trace()
        return v



    def full_lai(self, w,x,z):
        ''' full Loss Augmented Inference
         y_ = arg min <w|-psi(x,y_)> - loss(y,y_) '''
            
        ## combine all weight functions
        weight_function = MetaLaplacianFunction(
            np.asarray(w)[self.indices_laplacians],
            self.laplacian_functions,
            )
                
        ## loss type
        if self.loss_type=='anchor':
            nlabel = self.nlabel
            nnode = len(z[0])
            ztilde = (1. - np.asarray(z)) / (nlabel - 1.0)
            loss = {'data': ztilde}
            loss_weight = (self.nlabel - 1.0) / float(nlabel*nnode)
            L_loss = None
        elif self.loss_type=='laplacian':
            loss = None
            loss_weight = None
            L_loss = (-1.0)*laplacian_loss(z, mask=self.mask)
        else:
            raise Exception('did not recognize loss type')
            sys.exit(1)
        
        ## loss function        
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
            additional_laplacian=L_loss,
            **self.rwparams
            )
            
        return y_
        
    def worst_loss_inference(self, x,z):
        nlabel = len(self.labelset)
        y_ = (1-z)/float(nlabel-1)
        return y_
        
    def best_segmentation_inference(self, w,x):
        anchor_api = MetaAnchor(
            prior=self.prior,
            prior_models=self.prior_models,
            prior_weights=np.asarray(w)[self.indices_priors],
            loss=None,
            loss_weight=None,
            image=x,
            )
    
        ## combine all weight functions
        weight_function = MetaLaplacianFunction(
            np.asarray(w)[self.indices_laplacians],
            self.laplacian_functions,
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

                
    def compute_aci(self,w,x,z,y0):
        ''' annotation consistent inference'''
        
        weight_function = MetaLaplacianFunction(
            np.asarray(w)[self.indices_laplacians],
            self.laplacian_functions,
            )
        
        ## combine all prior models
        anchor_api = MetaAnchor(
            prior=self.prior,
            prior_models=self.prior_models,
            prior_weights=np.asarray(w)[self.indices_priors],
            image=x,
            )
        
        ## annotation consistent inference
        y = rwsegment.segment(
            x, 
            anchor_api,
            seeds=self.seeds,
            weight_function=weight_function,
            return_arguments=['y'],
            ground_truth=z,
            ground_truth_init=y0,
            **self.rwparams
            )
        
                
#-------------------------------------------------------------------------------
##------------------------------------------------------------------------------
def laplacian_loss(ground_truth, mask=None):
    from scipy import sparse
    
    size = ground_truth[0].size
    if mask is None:
        gt = ground_truth
        npix = size
    else:
        npix = np.sum(mask[0])
        gt = ground_truth*mask

    nlabel = len(ground_truth)
            
    ## TODO: max loss is with uniform probability
    weight = 1.0/float((nlabel-1)*npix)

    A_blocks = []
    for l2 in range(nlabel):
        A_blocks_row = []
        for l11 in range(l2):
            A_blocks_row.append(sparse.coo_matrix((size,size)))
        for l12 in range(l2,nlabel):
            A_blocks_row.append(
                sparse.spdiags(1.0*np.logical_xor(gt[l12],gt[l2]),0,size,size))
        A_blocks.append(A_blocks_row)
    A_loss = sparse.bmat(A_blocks)
     
    #import ipdb; ipdb.set_trace() ## not working  
    # A_loss = sparse.bmat([
    #     [sparse.coo_matrix((size,size)) for l11 in range(l2)] +
    #     [sparse.spdiags(1.0*(gt[l12]&(~gt[l2])),0,size,size) \
    #         for l12 in range(l2,nlabel)] \
    #     for l2 in range(nlabel)
    #     ])
        
    A_loss = A_loss + A_loss.T
    D_loss = np.asarray(A_loss.sum(axis=0)).ravel()
    L_loss = sparse.spdiags(D_loss,0,*A_loss.shape) - A_loss

    return -weight*L_loss.tocsr()
