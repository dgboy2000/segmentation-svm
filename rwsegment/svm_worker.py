import numpy as np
import gc

from mpi4py import MPI
import utils_logging
logger = utils_logging.get_logger('svm_worker',utils_logging.DEBUG)

from struct_svm import DataContainer

class SVMWorker(object):
    def __init__(self,svm_rw_api):
        self.api = svm_rw_api
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        self.cache_psi = {}
        
        
    def do_mvc(self,data):
        nproc = self.comm.Get_size()
        inds,w,S = data
        
        ys = []
        # ntrain = len(S)
        # for i in range(ntrain):
        for i,ind in enumerate(inds):
            # if (np.mod(i, nproc-1) + 1) != self.rank:
                # continue
                
            logger.debug('worker #{}: MVC on sample #{}'\
                .format(self.rank, ind))
            
            x,z = S[i]
            y_ = self.api.compute_mvc(w,x.data,z.data, exact=True)
            ys.append((ind, DataContainer(y_)))
            
        ## send data
        for i, y_ in ys:
            logger.debug('worker #{} sending back MVC for sample #{}'\
                .format(self.rank, i))
            self.comm.send(y_, dest=0, tag=i)
            
        gc.collect()
            
    def do_psi(self,data):
        nproc = self.comm.Get_size()
        inds, S,ys = data
        ntrain = len(S)
        psis = []
        # for i in range(ntrain):
        for i,ind in enumerate(inds):
            # if (np.mod(i, nproc-1) + 1) != self.rank:
                # continue
                
            logger.debug('worker #{}: PSI for sample #{}'\
                .format(self.rank, ind))
            
            x,z = S[i]
            if ys is None:
                psi = self.api.compute_psi(x.data, z.data)
            else:
                psi = self.api.compute_psi(x.data, ys[i].data)
                
            psis.append((ind,psi))
            
        ## send data
        for i,psi in psis:
            logger.debug('worker #{} sending back PSI for sample #{}'\
                .format(self.rank, i))
            self.comm.send(psi, dest=0, tag=i)
        
        gc.collect()
        
    def work(self,):
        while True:
            logger.debug('worker #{} about to receive next task'.format(self.rank))
            # task,data = self.comm.bcast(None,root=0)
            task,data = self.comm.recv(None,source=0)
            
            nproc = self.comm.Get_size()
            
            if task=='mvc':
                self.do_mvc(data)
            elif task=='psi':
                self.do_psi(data)
            elif task=='stop':
                logger.info('worker #{} received kill signal. Stopping.'\
                    .format(self.rank))
                break
            else:
                logger.fatal('worker #{} did not recognize task: {}'\
                    .format(self.rank, task))
                raise Exception('did not recognize task: {}'.format(task))
                sys.exit(0)
                
            gc.collect()