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
        self.laplacian_functions = laplacian_functions
 
        self.loss_type = kwargs.pop('loss_type','squareddiff')
        logger.info('using loss type: {}'.format(self.loss_type)) 
        self.approx_aci = kwargs.pop('approx_aci', False)
        self.loss_factor = float(kwargs.pop('loss_factor', 1.0))
        logger.info('loss is scaled by {:.3}'.format(self.loss_factor))       

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
        
        self.seeds = seeds
        self.rwparams = rwparams
        self.immask = seeds<0
        self.mask = np.asarray([self.immask.ravel() for i in range(self.nlabel)])
        self.prior_mask = np.zeros(seeds.shape, dtype=bool)
        self.prior_mask.flat[self.prior['imask']] = 1
        self.maskinds = np.arange(seeds.size).reshape(seeds.shape)
               
    def compute_loss(self,z,y_, **kwargs):
        islices = kwargs.pop('islices',None)
        if islices is not None:
            mask = [self.immask[islices].ravel() for i in range(len(self.labelset))]
        else:
            mask = self.mask

        if np.sum(y_<-1e-6) > 0:
            miny = np.min(y_)
            logger.warning('negative (<-1e-6) values in y_. min = {:.3}'.format(miny))

        #self.use_ideal_loss = True

        #if self.use_ideal_loss:
        if self.loss_type in ['ideal', 'none']:
            loss = loss_functions.ideal_loss(z,y_,mask=mask)
        elif self.loss_type=='squareddiff':
            loss = loss_functions.anchor_loss(z,y_,mask=mask)
        elif self.loss_type=='laplacian':
            loss = loss_functions.laplacian_loss(z,y_,mask=mask)
        elif self.loss_type=='linear':
            loss = loss_function.linear_loss(z,y,mask=mask)
        else:
           raise Exception('wrong loss type')
           sys.exit(1)
        return loss*self.loss_factor

    ## psi  
    def compute_psi(self, x,y, **kwargs):
        ''' - sum(a){Ea(x,y)} '''
        islices = kwargs.pop('islices',None)
        imask = kwargs.pop('imask',None)
        iimask = kwargs.pop('iimask',None)
        if islices is not None:
            im = x
            seeds = self.seeds[islices]
            prior = {
                'data': np.asarray(self.prior['data'])[:,iimask],
                'imask': imask,
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
        return v

    def full_lai(self, w,x,z, switch_loss=False, iter=-1, **kwargs):
        ''' full Loss Augmented Inference
         y_ = arg min <w|-psi(x,y_)> - loss(y,y_) '''

        islices = kwargs.pop('islices',None)
        imask = kwargs.pop('imask',None)
        iimask = kwargs.pop('iimask',None)
        if islices is not None:   
            im = x
            seeds = self.seeds[islices]
            mask = [self.immask[islices].ravel() for i in range(len(self.labelset))]
            prior = {
                'data': np.asarray(self.prior['data'])[:,iimask],
                'imask':imask,
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
        
        loss_type = self.loss_type
        if loss_type in ['ideal', 'none']:
            pass
        elif loss_type=='squareddiff':
            loss, loss_weight = loss_functions.compute_loss_anchor(seg, mask=mask)
            loss_weight *= self.loss_factor
        elif loss_type=='laplacian':
            L_loss = - loss_functions.compute_loss_laplacian(seg, mask=mask) *\
                 self.loss_factor
        elif loss_type=='linear':
            addlin, linw = loss_functions.compute_loss_linear(seg, mask=mask)
            addlin *= linw * self.loss_factor
        else:
            raise Exception('did not recognize loss type {}'.format(loss_type))
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
            #laplacian_label_weights=,
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

    def compute_exact_aci(self,w,x,z,y0,**kwargs):
        islices = kwargs.pop('islices',None)
        iimask = kwargs.pop('iimask',None)
        imask = kwargs.pop('imask',None)
        if islices is not None:
            seeds = self.seeds[islices]
            mask = [self.immask[islices].ravel() for i in range(len(self.labelset))]
            prior = {
                'data': np.asarray(self.prior['data'])[:,iimask],
                'imask': imask,
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
            #laplacian_label_weights=,
            **self.rwparams
            )
        return y 
 
    def compute_approximate_aci2(self, w,x,z,y0,**kwargs):
        logger.info('using approximate aci')
        islices = kwargs.pop('islices',None)
        imask = kwargs.pop('imask',None)
        iimask = kwargs.pop('iimask',None)
        if islices is not None:
            seeds = self.seeds[islices]
            mask = [self.immask[islices].ravel() for i in range(len(self.labelset))]
            prior = {
                'data': np.asarray(self.prior['data'])[:,iimask],
                'imask': imask,
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
            #laplacian_label_weights=,
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
            #laplacian_label_weights=,
            **self.rwparams
            )
        y[:,icorrect] = y_[:,icorrect]
        #import ipdb; ipdb.set_trace()
        return y                

    def compute_approximate_aci(self, w,x,z,y0,**kwargs):
        logger.info("using approximate aci (Danny's)")
        islices = kwargs.pop('islices',None)
        imask = kwargs.pop('imask',None)
        iimask = kwargs.pop('iimask',None)
        if islices is not None:
            seeds = self.seeds[islices]
            mask = [self.immask[islices].ravel() for i in range(len(self.labelset))]
            prior = {
                'data': np.asarray(self.prior['data'])[:,iimask],
                'imask': imask,
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

        class GroundTruthAnchor(object):
            def __init__(self, anchor_api, gt, gt_weights):
                self.anchor_api = anchor_api
                self.gt = gt
                self.gt_weights = gt_weights
            def get_labelset(self): 
                return self.anchor_api.get_labelset()

            def get_anchor_and_weights(self, D, indices):
                anchor, weights = self.anchor_api.get_anchor_and_weights(D,indices)
                gt_weights = self.gt_weights[:,indices]
                gt = self.gt[:,indices]
                new_weights = weights + gt_weights
                new_anchor = (anchor * weights + gt*gt_weights) / new_weights
                return new_anchor, new_weights
                
        self.approx_aci_maxiter = 200
        self.approx_aci_maxstep = 1e-2
        z_weights = np.zeros(np.asarray(z).shape)
        z_label = np.argmax(z,axis=0)
        for i in range(self.approx_aci_maxiter):
            logger.debug("approx aci, iter={}".format(i))
    
            ## add ground truth to anchor api
            modified_api = GroundTruthAnchor(anchor_api, z, z_weights)

            ## inference
            y_ = rwsegment.segment(
                x, 
                modified_api,
                seeds=seeds,
                weight_function=weight_function,
                return_arguments=['y'],
                #laplacian_label_weights=,
                **self.rwparams
                )

            ## loss            
            #loss = self.compute_loss(z,y_, islices=islices)
            loss = loss_functions.ideal_loss(z,y_,mask=mask)
            logger.debug('loss = {}'.format(loss))
            if loss < 1e-8: 
                break
            
            #inc = np.where(z_label!=np.argmax(y_,axis=0))[0]
            #print len(inc), inc
            #import ipdb; ipdb.set_trace()
            ## uptade weights
            delta = np.max(y_ - y_[z_label, np.arange(y_.shape[1])], axis=0)
            delta = np.clip(delta, 0, self.approx_aci_maxstep)
            z_weights += delta

        return y_        


   

