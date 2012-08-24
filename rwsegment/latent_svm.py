import logging
import numpy as np

import utils_logging
logger = utils_logging.get_logger('struct_svm',utils_logging.DEBUG)

import struct_svm

class LatentSVM(object):
    def __init__(
            self, 
            loss_function, 
            psi, 
            most_violated_constraint,
            annotation_consistent_inference,
            **kwargs):
            
        self.user_loss = loss_function
        self.user_psi = psi
        self.user_mvc = most_violated_constraint
        self.user_aci = annotation_consistent_inference
        
        self.w0     = kwargs.pop('latent_w0',None)
        
        self.niter_max = kwargs.pop('latent_niter_max',100)
        self.epsilon = kwargs.pop('latent_epsilon',1e-3)
        self.epsilon = kwargs.pop('latent_w0', 1e-3)
        
        self.use_parallel = kwargs.pop('latent_use_parallel',False)
        
        self.svm_params = kwargs
            
    def psi(self,x,y):
        return self.user_psi(x,y)
            
    def _sequential_all_aci(self,w,xs,zs,y0s):
        for x,z in zip(xs,zs):
            ys.append(self.user_aci(w,x,z,y0))
            
    def all_annotation_consistent_inference(self,w,xs,zs,y0s):
        if self.use_parallel:
            return self._parallel_all_aci(w,xs,zs,y0s) ## TODO
        else:
            return self._sequential_all_aci(w,xs,zs,y0s)
            
            
            
    def train(self, training_set):

        ## load images
        ntrain = len(training_set)
        images = []
        hard_seg = []
        for x,z in training_set:
            images.append(x)
            hard_seg.append(z)
            
        # self.images = images
        # self.hard_seg = hard_seg
        
        ## initial w
        if self.w0 is None:
            logger.info("compute length of psi")
            nparam = len(self.psi(images[0], hard_seg[0]))
            w0 = np.ones(nparam)
            
        w = self.w0
        nparam = len(w)
        
        ## initial y
        # ys = [None for i in range(ntrain)]
        ys = hard_seg
        
        for iter in self.niter_max:
            
            ## annotation consistent inference
            ys = self.all_annotation_consistent_inference(w, images, hard_seg, ys)
            
            struct_trainint_set = [(x,y) for x,y in zip(images,ys)]
            
            ## convec struct svm
            struct_svm = struct_svm.StructSVM(
                struct_trainint_set,
                self.loss_function,
                self.psi,
                self.most_violated_constraint
                )
            w,xi = train()
               
            ## Stop condition
            # TODO: check it does the right thing
            if 