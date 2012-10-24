import logging
import numpy as np
import svm_worker

import utils_logging
logger = utils_logging.get_logger('latent_svm',utils_logging.DEBUG)

from struct_svm import StructSVM

class LatentSVM(object):
    def __init__(
            self, 
            loss_function, 
            psi, 
            most_violated_constraint,
            annotation_consistent_inference,
            **kwargs):
        
        ## store user function 
        self.user_loss = loss_function
        self.user_psi  = psi
        self.user_mvc  = most_violated_constraint
        self.user_aci  = annotation_consistent_inference
        self.struct_params = kwargs ## pass parameters to struct SVM
        
        ## set latent SVM specific parameters
        self.niter_max = kwargs.pop('latent_niter_max',100)
        self.epsilon = kwargs.pop('latent_epsilon',1e-3)

        ## parallel stuff
        self.use_parallel = kwargs.pop('latent_use_parallel',False) 
            
    def sequential_all_aci(self,w,xs,zs,metas,y0s):
        ys = []
        for x,z,y0,meta in zip(xs,zs,y0s,metas):
            ys.append(self.user_aci(w,x,z,y0,**meta))
        return ys
        
    def parallel_all_aci(self,w,xs,zs,metas,y0s):
        ws = [w for s in range(len(xs))]
        ys = svm_worker.broadcast('aci', ws, xs, zs, y0s, metas)
        return ys

        '''
        comm = mpi.COMM
        size = mpi.SIZE
        logger.debug('MPI size={}'.format(size)) 
        opts = {}
                
        metadata = [{'islices':x.islices,'iimask':x.iimask} for x in xs]

        ntrain = len(xs)
        indices = np.arange(ntrain)
        for n in range(1,size):       
            inds = indices[np.mod(indices,size-1) == (n-1)]
            comm.send(('aci',len(inds), opts), dest=n)
            for i in inds:
                comm.send((i,w,xs[i],y0s[i],zs[i], metadata[i]), dest=n)
                

        ys = []
        for i in range(ntrain):
            source_id = np.mod(i,comm.Get_size()-1) + 1
            ys.append( 
                comm.recv(source=source_id,tag=i),
                )
        return ys
        '''

    def all_aci(self,w,xs,zs,metas,y0s=None):
        if y0s is None:
            y0s = zs
        if self.use_parallel:
            return self.parallel_all_aci(w,xs,zs,metas,y0s)
        else:
            return self.sequential_all_aci(w,xs,zs,metas,y0s)
       
    def print_w(self,w):
        wstr = '[{}]'.format(', '.join('{:.04}'.format(val) for val in w))
        return wstr

    def train(self, images, annotations, metadata=None, **kwargs):
        ## load images
        ntrain = len(images)
        xs = images
        zs = annotations
        if metadata is None:
            metas = [None for i in range(ntrain)]
        else: metas = metadata            

        ## initial w
        w0 = kwargs.pop('w0',1)
        w = w0
       
        ## initial latent variable
        y0s = kwargs.pop('init_latents', None)
       
        ## initial objective
        svm_objective0 = 0
        
        for niter in range(1,self.niter_max+1):
            logger.debug('iteration (latent) #{}'.format(niter))
            
            ## annotation consistent inference
            logger.debug('annotation consistent inference (with w={})'\
                         .format(self.print_w(w)))
            ys = self.all_aci(w, xs, zs, metas, y0s=y0s)
           
            ## convec struct svm
            logger.debug('start struct svm')
            struct_svm = StructSVM(
                loss_function=self.user_loss,
                psi=self.user_psi,
                most_violated_constraint=self.user_mvc,
                **self.struct_params
                )
            w,xi,struct_info = struct_svm.train(xs, ys, metadata=metas, w=w, **kwargs)

            ## Stop condition
            svm_objective = struct_svm.objective(w,xi)
            strw = self.print_w(w)
            if (svm_objective - svm_objective0) < self.epsilon:
                logger.debug('latent svm stopping with: w={}, xi={:.04}'\
                             .format(strw, float(xi)))
                info = {'converged': True}
                return w, xi, ys, info

            ## end current iteration 
            logger.debug('done iteration #{}, with: w={}, xi={:.04}'.format(niter, strw, float(xi)))
            logger.debug('objective={:.04}'.format(svm_objective))
            svm_objective0 = svm_objective

        else:
            info = {'converged': False}
            logger.info("max number of iterations reached ({})".format(self.niter_max))
            return w, xi, ys, info

     ## end LatentSVM
            
            
