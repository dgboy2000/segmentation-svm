import logging
import numpy as np
import mpi

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
            
        self.user_loss = loss_function
        self.user_psi = psi
        self.user_mvc = most_violated_constraint
        self.user_aci = annotation_consistent_inference
        
        
        self.niter_max = kwargs.pop('latent_niter_max',100)
        self.latent_C = kwargs.pop('latent_C',10)
        self.epsilon = kwargs.pop('latent_epsilon',1e-3)
        
        self.use_parallel = kwargs.pop('latent_use_parallel',False) 
        self.svm_params = kwargs
        self.one_iteration = kwargs.pop('one_iteration',False)
        self.struct_params = kwargs
            
    def psi(self,x,y, **kwargs):
        return self.user_psi(x,y,**kwargs)
    
    def loss_function(self,z,y, **kwargs):
        return self.user_loss(z,y, **kwargs)    
    
    def most_violated_constraint(self,w,x,z, **kwargs):
        return self.user_mvc(w,x,z,**kwargs)
    
    def _sequential_all_aci(self,w,xs,zs,y0s):
        ys = []
        for x,z,y0 in zip(xs,zs,y0s):
            ys.append(self.user_aci(w,x,z,y0, islices=x.islices, iimask=x.iimask))
        return ys
        
    def _parallel_all_aci(self,w,xs,zs,y0s):
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
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
        
    def all_annotation_consistent_inference(self,w,xs,zs,y0s):
        if self.use_parallel:
            return self._parallel_all_aci(w,xs,zs,y0s) ## TODO
        else:
            return self._sequential_all_aci(w,xs,zs,y0s)
            
    def svm_objective(self,w,xi):
        ## TODO: check average cutting place is valid for latent
        obj = 0.5 * np.sum(np.square(w)) + self.latent_C * xi
        return obj
        
    def train(self, training_set, **kwargs):
    
        ## load images
        ntrain = len(training_set)
        images = []
        hard_seg = []
        for x,z in training_set:
            images.append(x)
            hard_seg.append(z)
            
        ## initial w
        w0 = kwargs.pop('w0',None)
        if w0 is None:
            logger.info("compute length of psi")
            nparam = len(self.psi(images[0], hard_seg[0]))
            w0 = np.ones(nparam)
            
        w = w0
        nparam = len(w)
       
        ## initial y
        ys = kwargs.pop('ys', hard_seg)
       
        ## current iteration
        niter0 = kwargs.pop('niter0', 1)
 
        ## initial objective
        obj = 0
        
        for niter in range(niter0, self.niter_max+1):
            logger.debug('iteration (latent) #{}'.format(niter))
            
            ## annotation consistent inference
            strw = ' '.join('{:.3}'.format(val) for val in np.asarray(w))
            logger.debug('annotation consistent inference (with w = [{}])'.format(strw))
            ys = self.all_annotation_consistent_inference(w, images, hard_seg, ys)
           
            ## build updated training set for struct svm
            struct_training_set = [(x,y) for x,y in zip(images,ys)]
 
            ## convec struct svm
            logger.debug('struct svm')
            struct_svm = StructSVM(
                struct_training_set,
                self.loss_function,
                self.psi,
                self.most_violated_constraint,
                **self.struct_params
                )
            w,xi,struct_info = struct_svm.train(w=w)

            info = {'ys': ys, 'niter': niter}
 
            ## Stop condition
            ## TODO: check it does the right thing
            if (self.svm_objective(w,xi) - obj) < self.epsilon:
                strw = ' '.join('{:.3}'.format(float(val)) for val in w)
                logger.debug('latent svm stopping with: w=[{}],xi={}'\
                    .format(strw,float(xi)))
                info['stopped'] = True
                return w,xi,info
            else:
                strw = ' '.join('{:.3}'.format(float(val)) for val in w)
                logger.debug('done iteration #{}, with: w=[{}],xi={}'\
                    .format(niter, strw,float(xi)))
                obj = self.svm_objective(w,xi)
                info['stopped'] = False
                if self.one_iteration:
                    return w,xi,info
                
     ## end LatentSVM
            
            
