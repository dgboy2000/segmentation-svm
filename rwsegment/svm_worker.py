from mpi4py import MPI

class SVMWorker(object):
    def __init__(self,svm_rw_api):
        self.api = svm_rw_api
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
    def work(self,):
        while True:
            w,x,z = self.comm.scatter(None,root=0)
            if len(w)==0:
                print 'Process #{}: received kill signal. Stopping.'\
                    .format(self.rank)
                break
            y_ = self.api.compute_mvc(w,x,z,exact=True)
            self.comm.gather(y_,root=0) 
            