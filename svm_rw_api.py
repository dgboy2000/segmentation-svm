import numpy as np
from rwsegment import rwsegment
from rwsegment.rwsegment import BaseAnchorAPI
from rwsegment import loss_functions
reload(rwsegment)
reload(loss_functions)

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
        
    def get_anchor_and_weights(self, D, indices):
        all_anchor = 0
        all_weights = 0
        
        ## prior models
        for imodel, model in enumerate(self.prior_models):
            api = model(
                self.prior, 
                anchor_weight=self.prior_weights[imodel],
                image=self.image,
                )
            anchor, weights = api.get_anchor_and_weights(D, indices)
            all_anchor  = all_anchor  + weights * anchor
            all_weights = all_weights + weights
           
        ## loss
        if self.loss is not None:
            all_anchor  = all_anchor + \
                self.loss_weight * self.loss[:,indices]
            all_weights += self.loss_weight
       
        if np.max(all_weights) < 1e-10:
             all_anchor = np.zeros(all_anchor.shape)
             all_weights = np.zeros(all_weights.shape)
        else: 
             all_anchor = all_anchor / all_weights
        return all_anchor, all_weights 



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
        
        self.approx_aci = kwargs.pop('approx_aci', False)

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
        self.immask = seeds<0
        self.mask = np.asarray([self.immask.ravel() for i in range(self.nlabel)])
        self.prior_mask = np.zeros(seeds.shape, dtype=bool)
        self.prior_mask.flat[self.prior['imask']] = 1
        self.maskinds = np.arange(seeds.size).reshape(seeds.shape)
       
        # rescale loss 
        self.loss_factor = float(kwargs.pop('loss_factor', 1.0))
        logger.warning('loss is scaled by {:.3}'.format(self.loss_factor))
        
    def compute_loss(self,z,y_, **kwargs):
        islices = kwargs.pop('islices',None)
        if islices is not None:
            mask = [self.immask[islices].ravel() for i in range(len(self.labelset))]
        else:
            mask = self.mask

        if np.sum(y_<-1e-6) > 0:
            miny = np.min(y_)
            logger.warning('negative (<-1e-6) values in y_. min = {:.3}'.format(miny))

        self.use_ideal_loss = True

        if self.use_ideal_loss:
            loss = loss_functions.ideal_loss(z,y_,mask=mask)
        elif self.loss_type=='anchor':
            loss = loss_functions.anchor_loss(z,y,mask=mask)
        elif self.loss_type=='laplacian':
            loss = loss_functions.laplacian_loss(z,y,mask=mask)
        else:
           raise Exception('wrong loss type')
           sys.exit(1)
        return loss*self.loss_factor

    ## psi  
    def compute_psi(self, x,y, **kwargs):
        ''' - sum(a){Ea(x,y)} '''

        islices = kwargs.pop('islices',None)
        iimask = kwargs.pop('iimask',None)
        if islices is not None:
            im = x
            seeds = self.seeds[islices]
            prior = {
                'data': np.asarray(self.prior['data'])[:,iimask],
                'imask': np.digitize(self.prior['imask'][iimask], self.maskinds[islices].ravel()),
                'variance': np.asarray(self.prior['variance'])[:,iimask],
                'labelset': self.labelset,
                } 
            if 'intensity' in self.prior: prior['intensity'] = self.prior['intensity']
            seg = y 
        else:
            im = x
            seeds = self.seeds
            prior = self.prior
            seg = y

       
        ## normalizing by the approximate mask size
        # normalize = float(nnode)/100.0
        normalize = 1.0
        
        ## energy value for each weighting function
        v = []
        for wf in self.laplacian_functions:
            
            v.append( 
                rwsegment.energy_rw(
                    im, seg,
                    seeds=seeds,
                    weight_function=wf,
                    **self.rwparams
                )/normalize)
                
        ## energy for each prior models
        for model in self.prior_models:
            anchor_api = model( 
                prior, 
                anchor_weight=1.0,
                image=im, # intensity prior needs an image
                )
            
            v.append(
                rwsegment.energy_anchor(
                    im, seg, anchor_api,
                    seeds=seeds,
                    weight_function=wf,
                    **self.rwparams
                )/normalize)
            
        if v[0]==0:
            import ipdb; ipdb.set_trace()
        return v



    def full_lai(self, w,x,z, switch_loss=False, iter=-1, **kwargs):
        ''' full Loss Augmented Inference
         y_ = arg min <w|-psi(x,y_)> - loss(y,y_) '''

        islices = kwargs.pop('islices',None)
        iimask = kwargs.pop('iimask',None)
        if islices is not None:   
            im = x
            seeds = self.seeds[islices]
            mask = [self.immask[islices].ravel() for i in range(len(self.labelset))]
            prior = {
                'data': np.asarray(self.prior['data'])[:,iimask],
                'imask': np.digitize(self.prior['imask'][iimask], self.maskinds[islices].ravel()),
                'variance': np.asarray(self.prior['variance'])[:,iimask],
                'labelset': self.labelset,
                }
            if 'intensity' in self.prior: prior['intensity'] = self.prior['intensity']
            seg = z
        else:
            im = x
            mask = self.mask
            seeds = self.seeds
            prior = self.prior
            seg = z
 
           
        ## combine all weight functions
        weight_function = MetaLaplacianFunction(
            np.asarray(w)[self.indices_laplacians],
            self.laplacian_functions,
            )
                
        ## loss type
        addlin      = None
        loss        = None
        loss_weight = None
        L_loss      = None
        
        if switch_loss:
            loss_type = 'approx'
        else:
            loss_type = self.loss_type

        if iter==0:
            loss_type = 'anchor'

        if loss_type=='none':
            pass
        elif loss_type=='anchor':
            loss, loss_weight = loss_functions.compute_loss_anchor(seg, mask=mask)
        elif loss_type=='laplacian':
            L_loss = loss_functions.compute_loss_laplacian(seg, mask=mask)
        elif loss_type=='approx':
            nnode = len(z[0])
            addlin = 1./float(nnode)*z
            addlin *= self.loss_factor
        else:
            raise Exception('did not recognize loss type')
            sys.exit(1)
        
        ## loss function        
        anchor_api = MetaAnchor(
            prior=prior,
            prior_models=self.prior_models,
            prior_weights=np.asarray(w)[self.indices_priors],
            loss=loss,
            loss_weight=loss_weight,
            image=im,
            )
        
        ## best y_ most different from y
        y_ = rwsegment.segment(
            im, 
            anchor_api,
            seeds=seeds,
            weight_function=weight_function,
            return_arguments=['y'],
            additional_laplacian=L_loss,
            additional_linear=addlin,
            **self.rwparams
            )
            
        return y_
        
   
    def compute_mvc(self,w,x,z,exact=True, switch_loss=False, **kwargs):
        return self.full_lai(w,x,z, switch_loss=switch_loss, **kwargs)
                        
    def compute_aci(self,*args, **kwargs):
        ''' annotation consistent inference'''
        if self.approx_aci:
            return self.compute_approximate_aci(*args, **kwargs)
        else:
            return self.compute_exact_aci(*args, **kwargs)

    def compute_exact_aci(self, w,x,z,y0,**kwargs):
        islices = kwargs.pop('islices',None)
        iimask = kwargs.pop('iimask',None)
        if islices is not None:
            seeds = self.seeds[islices]
            mask = [self.immask[islices].ravel() for i in range(len(self.labelset))]
            prior = {
                'data': np.asarray(self.prior['data'])[:,iimask],
                'imask': np.digitize(self.prior['imask'][iimask], self.maskinds[islices].ravel()),
                'variance': np.asarray(self.prior['variance'])[:,iimask],
                'labelset': self.labelset,
                }
            if 'intensity' in self.prior: prior['intensity'] = self.prior['intensity']
        else:
            mask = self.mask
            seeds = self.seeds
            prior = self.prior
        
        weight_function = MetaLaplacianFunction(
            np.asarray(w)[self.indices_laplacians],
            self.laplacian_functions,
            )
        
        ## combine all prior models
        anchor_api = MetaAnchor(
            prior=prior,
            prior_models=self.prior_models,
            prior_weights=np.asarray(w)[self.indices_priors],
            image=x,
            )
        
        ## annotation consistent inference
        y = rwsegment.segment(
            x, 
            anchor_api,
            seeds=seeds,
            weight_function=weight_function,
            return_arguments=['y'],
            ground_truth=z,
            ground_truth_init=y0,
            **self.rwparams
            )
        return y 
 
    def compute_approximate_aci(self, w,x,z,y0,**kwargs):
        logger.info('using approximate aci')
        islices = kwargs.pop('islices',None)
        iimask = kwargs.pop('iimask',None)
        if islices is not None:
            seeds = self.seeds[islices]
            mask = [self.immask[islices].ravel() for i in range(len(self.labelset))]
            prior = {
                'data': np.asarray(self.prior['data'])[:,iimask],
                'imask': np.digitize(self.prior['imask'][iimask], self.maskinds[islices].ravel()),
                'variance': np.asarray(self.prior['variance'])[:,iimask],
                'labelset': self.labelset,
                }
            if 'intensity' in self.prior: prior['intensity'] = self.prior['intensity']
        else:
            mask = self.mask
            seeds = self.seeds
            prior = self.prior
        
        weight_function = MetaLaplacianFunction(
            np.asarray(w)[self.indices_laplacians],
            self.laplacian_functions,
            )
        
        ## combine all prior models
        anchor_api = MetaAnchor(
            prior=prior,
            prior_models=self.prior_models,
            prior_weights=np.asarray(w)[self.indices_priors],
            image=x,
            )

        ## unconstrained inference
        y_ = rwsegment.segment(
            x, 
            anchor_api,
            seeds=seeds,
            weight_function=weight_function,
            return_arguments=['y'],
            **self.rwparams
            )

        ## fix correct labels
        gt = np.argmax(z,axis=0)
        icorrect = np.argmax(y_,axis=0)==gt
        seeds_correct = -np.ones(seeds.shape, dtype=int)
        seeds_correct.flat[icorrect] = self.labelset[gt[icorrect]]

        ## annotation consistent inference
        #import ipdb; ipdb.set_trace()
        y = rwsegment.segment(
            x, 
            anchor_api,
            seeds=seeds_correct,
            weight_function=weight_function,
            return_arguments=['y'],
            ground_truth=z,
            ground_truth_init=y0,
            seeds_prob=y_,
            **self.rwparams
            )
        y[:,icorrect] = y_[:,icorrect]
        #import ipdb; ipdb.set_trace()
        return y                
#-------------------------------------------------------------------------------
##------------------------------------------------------------------------------
'''
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
'''

