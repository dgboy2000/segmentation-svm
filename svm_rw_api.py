import numpy as np
import sys
from rwsegment import rwsegment
from rwsegment.rwsegment import BaseAnchorAPI
from rwsegment import loss_functions
reload(rwsegment)
reload(loss_functions)

from rwsegment import utils_logging
logger = utils_logging.get_logger('svm_rw_api',utils_logging.DEBUG)

import svm_rw_functions
reload(svm_rw_functions)
              
class SVMRWMeanAPI(object):
    def __init__(
            self, 
            laplacian_functions,
            prior_models,
            seeds,
            labelset,
            rwparams,
            **kwargs):
        
        self.laplacian_functions = laplacian_functions
        self.prior_models  = prior_models
        self.labelset = labelset
        self.weights = np.ones(self.anchor.shape)
        self.nlabel   = len(labelset)

        self.loss_type = kwargs.pop('loss_type','squareddiff')
        self.approx_aci = kwargs.pop('approx_aci', False)
        self.loss_factor = float(kwargs.pop('loss_factor', 1.0))
        logger.info('using loss type: {}'.format(self.loss_type)) 
        logger.info('loss is scaled by {:.3}'.format(self.loss_factor))       

        self.seeds = seeds
        self.mask = seeds<0
        self.rwparams = rwparams
               
    def compute_loss(self,z,y_, **kwargs):
        islices = kwargs.pop('islices',slice(None))
        flat_mask = [self.mask[islices].ravel() for s in range(self.nlabel)]

        if np.sum(y_<-1e-6) > 0:
            miny = np.min(y_)
            logger.warning('negative (<-1e-6) values in y_. min = {:.3}'.format(miny))

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
    def compute_psi(self, x, y, **kwargs):
        ''' - sum(a){Ea(x,y)} '''
        islices = kwargs.pop('islices', slice(None))

        ## energy value for each weighting function
        psi = []
        for laplacian_function in self.laplacian_functions:
            name = laplacian_function['name']
            function = laplacian_function['func']
            psi.extend( 
                rwsegment.energy_rw(
                    x, y,
                    seeds=seeds[islices],
                    laplacian_function=function,
                    **self.rwparams))
                
        ## energy for each prior models
        for model in self.prior_model:
            name = model['name']
            anchor_api = svm_rw_functions.AnchorReslice(
               x.shape, model['api'], islices=islices)
            psi.extend(
                rwsegment.energy_anchor(
                    x, y, anchor_api,
                    seeds=seeds[islices],
                    **self.rwparams))
        return psi

    def full_lai(self, w,x,z, switch_loss=False, iter=-1, **kwargs):
        ''' full Loss Augmented Inference '''
        labelset = self.labelset
        nlabel = len(labelset)
        nvar   = len(z[0])
        islices = kwargs.pop('islices', slice(None))
        mask  = self.mask[islices].ravel()
        flatmask = [mask for s in range(self.nlabel)]

        ## loss type
        addlin      = None
        loss_anchor  = []
        loss_weight = []
        L_loss      = None
        
        loss_type = self.loss_type
        if loss_type in ['ideal', 'none']:
            pass
        elif loss_type=='squareddiff':
            loss, loss_weight = loss_functions.compute_loss_anchor(seg, mask=flatmask)
            loss_anchor = [BaseAnchorApi(
               np.arange(nvar), loss)]
        elif loss_type=='laplacian':
            L_loss = - loss_functions.compute_loss_laplacian(seg, mask=flatmask) *\
                 self.loss_factor
        elif loss_type=='linear':
            addlin, linw = loss_functions.compute_loss_linear(seg, mask=flatmask)
            addlin *= linw * self.loss_factor
        else:
            raise Exception('did not recognize loss type {}'.format(loss_type))
            sys.exit(1)

        ## laplacian functions
        nlaplacian = len(self.laplacian_functions]
        lweights = w[:nlabel*nlaplacian]
        lfuncs = [f['func' for f in self.laplacian_functions]
        laplacian_function = svm_rw_functions.LaplacianWeights(
            nlabel,lfuncs, lweights) 
 
        ## anchors
        amodels  = []
        for model in self.prior_models:
            amodels.append(svm_rw_functions.AnchorReslice(
                x.shape, model['api'], islice=islice))
        amodels.append(loss_anchor)
        aweights = w[nlabel*nlaplacian:] + loss_weight
        anchor_api = svm_rw_functions.MetaAnchorApi(amodels, aweights)
        
        ## best y_ most different from y
        y_ = rwsegment.segment(
            im, 
            anchor_api,
            labelset,
            seeds=seeds[islices],
            laplacian_function=laplacian_function,
            return_arguments=['y'],
            additional_laplacian=L_loss,
            additional_linear=addlin,
            **self.rwparams)
            
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
        labelset = self.labelset
        nlabel = len(labelset)
        islices = kwargs.pop('islices',slice(None))
         
        ## laplacian functions
        nlaplacian = len(self.laplacian_functions]
        lweights = w[:nlabel*nlaplacian]
        lfuncs = [f['func' for f in self.laplacian_functions]
        laplacian_function = svm_rw_functions.LaplacianWeights(
            nlabel,lfuncs, lweights) 
 
        ## anchors
        amodels  = []
        for model in self.prior_models:
            amodels.append(svm_rw_functions.AnchorReslice(
                x.shape, model['api'], islice=islice))
        aweights = w[nlabel*nlaplacian:]
        anchor_api = svm_rw_functions.MetaAnchorApi(amodels, aweights)
             
        ## annotation consistent inference
        y = rwsegment.segment(
            x, 
            anchor_api,
            labelset,
            seeds=seeds[islices],
            laplacian_function=weight_function,
            return_arguments=['y'],
            ground_truth=z,
            ground_truth_init=y0,
            **self.rwparams
            )
        return y 
 
    def compute_approximate_aci2(self, w,x,z,y0,**kwargs):
        logger.info('using approximate aci')
        labelset = self.labelset
        nlabel = len(labelset)
        islices = kwargs.pop('islices',slice(None))
         
        ## laplacian functions
        nlaplacian = len(self.laplacian_functions]
        lweights = w[:nlabel*nlaplacian]
        lfuncs = [f['func' for f in self.laplacian_functions]
        laplacian_function = svm_rw_functions.LaplacianWeights(
            nlabel,lfuncs, lweights) 
 
        ## anchors
        amodels  = []
        for model in self.prior_models:
            amodels.append(svm_rw_functions.AnchorReslice(
                x.shape, model['api'], islice=islice))
        aweights = w[nlabel*nlaplacian:]
        anchor_api = svm_rw_functions.MetaAnchorApi(amodels, aweights)
 
        ## unconstrained inference
        y_ = rwsegment.segment(
            x, 
            anchor_api,
            labelset,
            seeds=seeds,
            laplacian_function=weight_function,
            return_arguments=['y'],
            **self.rwparams
            )

        ## fix correct labels
        gt = np.argmax(z,axis=0)
        icorrect = np.argmax(y_,axis=0)==gt
        seeds_correct = -np.ones(seeds.shape, dtype=int)
        seeds_correct.flat[icorrect] = labelset[gt[icorrect]]

        ## annotation consistent inference
        #import ipdb; ipdb.set_trace()
        y = rwsegment.segment(
            x, 
            anchor_api,
            labelset,
            seeds=seeds_correct,
            labelset_function=weight_function,
            return_arguments=['y'],
            ground_truth=z,
            ground_truth_init=y0,
            seeds_prob=y_,
            **self.rwparams
            )
        y[:,icorrect] = y_[:,icorrect]
        return y                

    def compute_approximate_aci(self, w,x,z,y0,**kwargs):
        logger.info("using approximate aci (Danny's)")
        labelset = self.labelset
        nlabel = len(labelset)
        nvar = len(z[0])
        islices = kwargs.pop('islices',slice(None))
        mask  = self.mask[islices].ravel()
        flatmask = [mask for s in range(self.nlabel)]
         
        ## laplacian functions
        nlaplacian = len(self.laplacian_functions]
        lweights = w[:nlabel*nlaplacian]
        lfuncs = [f['func' for f in self.laplacian_functions]
        laplacian_function = svm_rw_functions.LaplacianWeights(
            nlabel,lfuncs, lweights) 
 
        ## anchors
        amodels  = []
        for model in self.prior_models:
            amodels.append(svm_rw_functions.AnchorReslice(
                x.shape, model['api'], islice=islice))
        aweights = w[nlabel*nlaplacian:]
        anchor_api = svm_rw_functions.MetaAnchorApi(amodels, aweights)
 
        self.approx_aci_maxiter = 200
        self.approx_aci_maxstep = 1e-2
        z_weights = np.zeros(np.asarray(z).shape)
        z_label = np.argmax(z,axis=0)
        for i in range(self.approx_aci_maxiter):
            logger.debug("approx aci, iter={}".format(i))
    
            ## add ground truth to anchor api
            gtmnodel = BaseAnchorAPI(np.arange(nvar), z, weights=z_weights)
            modified_api = svm_rw_functions.MetaAnchorApi(
                amodels + [gtmodel], aweights + [1] )

            ## inference
            y_ = rwsegment.segment(
                x, 
                modified_api,
                seeds=seeds,
                weight_function=weight_function,
                return_arguments=['y'],
                **self.rwparams
                )

            ## loss            
            loss = loss_functions.ideal_loss(z,y_,mask=flatmask)
            logger.debug('loss = {}'.format(loss))
            if loss < 1e-8: 
                break
            
            ## update weights
            delta = np.max(y_ - y_[z_label, np.arange(y_.shape[1])], axis=0)
            delta = np.clip(delta, 0, self.approx_aci_maxstep)
            z_weights += delta

        return y_        


   

