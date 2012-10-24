import numpy as np
import hashlib
import mpi

import logging
import svm_solver
reload(svm_solver)
import svm_worker
reload(svm_worker)
import utils_logging
logger = utils_logging.get_logger('struct_svm',utils_logging.DEBUG)

'''
class DataContainer(object):
    def __init__(self, data):
        self.data = data
    def __hash__(self):
        return id(self.data)
'''

class StructSVM(object):  
    '''
    "Cutting-plane training of structural SVMs"
    Machine Learning 2009
    Joachims, Thorsten
    Finley, Thomas
    Yu, Chun-Nam John

    '''
 
    def __init__(
            self, 
            #training_set, 
            loss_function, 
            psi, 
            most_violated_constraint,
            **kwargs):
        ''' 
            args:
                loss    : (scalar) = loss(y,y_)
                psi     : (vector) = psi(x,y)
                most_violated_constraint : y_ = most_violated_constraint(w,x,y)

        '''
        ## struct svm parameters 
        self.C = kwargs.pop('C',1.)
        self.epsilon = kwargs.pop('epsilon',1e-5)
        self.niter_max = kwargs.pop('nitermax',100)
        
        ## user provided functions 
        self.user_loss = loss_function
        self.user_psi  = psi
        self.user_mvc  = most_violated_constraint 
       
        #self.psi_cache  = {}
        self.psis_cache = {}
        #self.loss_cache = {}
        #self.mvc_cache  = {}      
  
        #self.psi_scale = kwargs.pop('psi_scale', 1.0)
        self.use_parallel = kwargs.pop('use_parallel', False)
        #nomosek = kwargs.pop('nomosek',False)

        #self.do_switch_loss = kwargs.pop('do_switch_loss', False)

    def parallel_all_mvc(self, w, xs, zs, metas, **kwargs):
        ws = [w for i in range(len(xs))]
        ys = svm_worker.broadcast('mvc', ws, xs, zs, metas, **kwargs)
        return ys

        '''
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        comm = mpi.COMM
        size = mpi.SIZE
        
        opts = kwargs         

        metadata = [{'islices':x.data.islices,'iimask':x.data.iimask} for x,y in self.S]

        ntrain = len(self.S)
        indices = np.arange(ntrain)
        for n in range(1,size):       
            inds = indices[np.mod(indices,size-1) == (n-1)]
            comm.send(('mvc',len(inds),opts), dest=n)
            for i in inds:
                x,z = self.S[i]
                comm.send((i,w,x,z, metadata[i]), dest=n)

        ys = []
        ntrain = len(self.S)
        for i in range(ntrain):
            source_id = np.mod(i,comm.Get_size()-1) + 1
            ys.append( 
                comm.recv(source=source_id,tag=i),
                )
        return ys
        '''
    
    def parallel_all_psi(self,xs,ys,metas):
        psis = svm_worker.broadcast('psi', xs, ys, metas)
        return psis
       
        '''
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        comm = mpi.COMM
        size = mpi.SIZE
       
        opts = {}
  
        metadata = [{'islices':x.data.islices,'iimask':x.data.iimask} for x,y in self.S]

        ## send training data and cutting plane
        ntrain = len(self.S)
        indices = np.arange(ntrain)
        for n in range(1,size):       
            inds = indices[np.mod(indices,size-1) == (n-1)]
            comm.send(('psi',len(inds), opts), dest=n)
            for ind in inds:
                x,z = self.S[ind]
                if ys is None:                
                    comm.send((ind,x,z,metadata[ind]), dest=n)
                else:
                    comm.send((ind,x,ys[ind], metadata[ind]), dest=n)
    
    
        ## get the psis back
        cond = 0
        ntrain = len(self.S)
        psis = []
        for i in range(ntrain):          
            ## recieve from workers
            source_id = np.mod(i,comm.Get_size()-1) + 1
            psi = comm.recv(source=source_id,tag=i)
            psis.append(psi)
            
        # return psis
        for i in range(ntrain):
            yield psis[i]
        '''           
       
    def psi(self, x,y, **kwargs):
        return self.user_psi(x,y,**kwargs)

        '''
        if x,y in self.psi_cache:
            return self.psi_cache[(x,y)]
        else:
            v = self.user_psi(x.data,y.data, iimask=x.data.iimask, islices=x.data.islices)
            self.psi_cache[(x,y)] = v
            return v
        '''
            
    def all_psi(self,xs,ys,metas):
        #if ys is None:
        #    obj = None
        #else:
        #    obj = ys[0]

        key = hashlib.sha1(ys[0])
        if key in self.psis_cache:
            for psi in self.psis_cache[key]:
                yield psi
            
        elif self.use_parallel:
            self.psis_cache[key] = []
            for psi in self.parallel_all_psi(xs,ys,metas):
                self.psis_cache[key].append(psi)
                yield psi
        else:
            self.psis_cache[key] = []
            for x,y,meta in zip(xs,ys,metas):
                psi = self.psi(x,y,**meta)
                self.psis_cache[key].append(psi)
                yield psi
            
    def loss(self,z,y,**kwargs):
        return self.user_loss(z,y, **kwargs)
        
    def mvc(self,w,x,z, **kwargs):
        return self.user_mvc(w,x,z,**kwargs)

    def all_mvc(self, w, xs, zs, metas, **kwargs):
        if self.use_parallel:
            return self.parallel_all_mvc(w, xs, zs, metas, **kwargs)
        else:
            return [self.mvc(w,x,z,**dict(meta,**kwargs)) for x,z,meta in zip(xs,zs,metas)]

    def stop_condition(self,w,xi,psis_gt,psis,losses):
        cond = 0
        ntrain = len(psis)
        for i in range(ntrain):
            cond += losses[i]
            for iw in range(len(w)):
                cond -= w[iw]*(psis[i][iw] - psis_gt[i][iw])
        cond /= float(ntrain)
        if cond <= xi + self.epsilon:
            return True
        else:
            return False
             
    def current_solution(self, W, gtpsis, w0=None, scale_only=False):
        solver = svm_solver.SVMSolver(
            self.C, 
            use_mosek=True, 
            scale_only=scale_only,
            )
        w,xi = solver.solve(W,gtpsis,w0=w0)
        return w,xi

    def objective(self,w,xi):
        obj = 0.5 * np.sum(np.square(w)) + self.C * xi
        return obj
 
    def print_vec(self,w):
        wstr = '[{}]'.format(', '.join('{:.04}'.format(val) for val in w))
        return wstr

    def print_psis(self,psis):
        psistr = '\n'.join(
            '[{}]'.format(
                ' '.join('{:.03}'.format(val) for val in psi)) \
                for psi in psis
            )
        return psistr

    def train(self, images, annotations, metadata=None, w=None, **kwargs):
        xs = images
        zs = annotations      
        if metadata is None:
            metas = [None for i in range(ntrain)]
        else: metas = metadata            
          
        ## compute psis of ground truth
        gtpsis = list(self.all_psi(xs,zs,metas))
        
        ## log psis
        logger.debug('ground truth psis: {}'.format(self.print_psis(gtpsis)))

        ## initial w
        if w is not None:
            w0 = [wi for wi in w]
        else: w0 = None
 
        ## test set for qp
        W = [] 
       
        #switch_loss = False 
        for niter in range(1, self.niter_max+1):
            logger.info("iteration (struct) #{}".format(niter))
            
            # garbage collect to save memory on each iteration
            import gc
            gc.collect()
            
            ## compute current solution (qp + constraints)
            logger.info("compute current solution")
        
            w,xi = self.current_solution(W, gtpsis, w0=w0, **kwargs)
 
            ## logging
            logger.debug("w={}, xi={:.04}".format(self.print_vec(w),xi))     
            logger.debug("objective={:.04}".format(self.objective(w,xi)))
       
            ## find most violated constraint
            logger.info("find most violated constraint")
            ys = self.all_mvc(w, xs, zs, metas)

            ## compute psis and losses:
            logger.debug('compute psis and losses for added constraints')
            psis = list(self.all_psi(xs,ys,metas))
            losses = [self.loss(z, y_, **meta) for z,y_,meta in zip(zs,ys,metas)]
            
            ## log psis and losses
            logger.debug('new psis: {}'.format(self.print_psis(psis)))
            logger.debug('new losses: {}'.format(self.print_vec(losses)))
            
            ## add to test set
            W.append({'psis': psis, 'losses': losses})
            
            ## stop condition
            logger.debug('compute stop condition')
            if self.stop_condition(w,xi,gtpsis,psis,losses):
                logger.info("stop condition reached, stopping")
                info = {'converged': True}
                break
        else:
            logger.info("max number of iterations reached ({})".format(self.niter_max))
            info = {'converged': False}

        return w, xi, info
    
