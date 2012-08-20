import numpy as np

from mpi4py import MPI
import utils_logging
logger = utils_logging.get_logger('worker_logger',utils_logging.DEBUG)

class SVMWorker(object):
    def __init__(self,svm_rw_api):
        self.api = svm_rw_api
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
    def work(self,):
        while True:
            logger.debug('worker #{} about to do broadcast'.format(self.rank))
            w,S = self.comm.bcast(None,root=0)
            # w,S = self.comm.scatter(None,root=0)
            if len(w)==0:
                print 'Process #{}: received kill signal. Stopping.'\
                    .format(self.rank)
                break
            ntrain = len(S)
            nproc = self.comm.Get_size()
            ys = []
            for i_s, s in enumerate(S):
                if (np.mod(i_s, nproc-1) + 1) != self.rank:
                    continue
                logger.debug('worker #{} about to process sample #{}'\
                    .format(self.rank, i_s))
                y_ = self.api.compute_mvc(w,s[0].data,s[1].data, exact=True)
                ys.append((i_s,y_))
            
            for i_s, y_ in ys:
                logger.debug('worker #{} about to send back sample #{}'\
                    .format(self.rank, i_s))
                self.comm.send(y_, dest=0, tag=i_s)
            # self.comm.gather(y_,root=0) 
            