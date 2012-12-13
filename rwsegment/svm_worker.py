import numpy as np
import gc
import mpi
import duald

import utils_logging
logger = utils_logging.get_logger('svm_worker',utils_logging.INFO)

#from struct_svm import DataContainer

def broadcast(task, *args, **kwargs):
    ''' send data to SVM worker '''
    comm = mpi.COMM
    nproc = mpi.SIZE
    ndata = len(args[0])
    all = np.arange(ndata)

    logger.info('main process sending {} data to workers'.format(ndata))
    ## send
    for iproc in range(1,nproc):
       some = all[np.mod(all, nproc-1) == (iproc-1)]
       comm.send((task, len(some), kwargs), dest=iproc) # send initial message
       for i in some:
           comm.send(tuple([i]+[arg[i] for arg in args]), dest=iproc)
     
    ## receive
    output = []
    for n in range(ndata):
        source_proc = np.mod(n, comm.Get_size()-1) + 1
        output.append(comm.recv(source=source_proc, tag=n))
    return output


class SVMWorker(object):
    def __init__(self,svm_rw_api):
        self.api = svm_rw_api
        # self.comm = MPI.COMM_WORLD
        self.comm = mpi.COMM
        self.rank = mpi.RANK
        logger.info('Starting worker {}'.format(self.rank))
        self.cache_psi = {}
        
    def do_duald(self, ndata, **kwargs):
        sub_list = self.receive_n_items(ndata)
        logger.info('worker #{}: processing #{} subproblems'\
            .format(self.rank, ndata))
        
        outdata = []
        for ind,nlabel,Pk,qk,gtk in sub_list:
            data = duald.solver_gt(nlabel,Pk,qk,gtk)
            outdata.append((ind,data))
            
        ## send back data
        for i, data in outdata:
            logger.debug('worker #{} sending back xk for subproblem #{}'\
                .format(self.rank, i))
            self.comm.send(data, dest=0, tag=i)
        gc.collect()
        
    def do_mvc(self, ndata, **kwargs):
        # Receive the specified number of tuples
        mvc_data = self.receive_n_items(ndata)
        ys = []
        for ind,w,x,z,metadata in mvc_data:
            logger.debug('worker #{}: MVC on sample #{}'\
                .format(self.rank, ind))

            y_ = self.api.compute_mvc(w,x,z, exact=True, **dict(metadata,**kwargs))
            ys.append((ind, y_))
            
        ## send data
        for i, y_ in ys:
            logger.debug('worker #{} sending back MVC for sample #{}'\
                .format(self.rank, i))
            self.comm.send(y_, dest=0, tag=i)
            
        gc.collect()
            
    def do_psi(self,ndata, **kwargs):
        # Receive the specified number of tuples
        psi_data = self.receive_n_items(ndata)
        psis = []
        
        for ind,x,y,metadata in psi_data:
            logger.debug('worker #{}: PSI for sample #{}'\
                .format(self.rank, ind))
            
            psi = self.api.compute_psi(x, y, **dict(metadata, **kwargs))    
            psis.append((ind,psi))
            
        ## send data
        for i,psi in psis:
            logger.debug('worker #{} sending back PSI for sample #{}'\
                .format(self.rank, i))
            self.comm.send(psi, dest=0, tag=i)
        
        gc.collect()
        
    def do_aci(self, ndata, **kwargs):        
        # Receive the specified number of tuples
        aci_data = self.receive_n_items(ndata)
        ys = []
        
        for ind,w,x,y0,z,metadata in aci_data:        
            logger.debug('worker #{}: ACI on sample #{}'\
                .format(self.rank, ind))
            
            y_ = self.api.compute_aci(w,x,z,y0,**dict(metadata, **kwargs))
            ys.append((ind, y_))
            
        ## send data
        for i, y_ in ys:
            logger.debug('worker #{} sending back ACI for sample #{}'\
                .format(self.rank, i))
            self.comm.send(y_, dest=0, tag=i)
            
        gc.collect()
        
    
    def receive_n_items(self, n):
        item_list = []
        for i in range(n):
            logger.debug('Worker {} about to receive data item {} of {}'\
                .format(self.rank, i+1, n))
            item = self.comm.recv(None,source=0)
            item_list.append(item)
        return item_list
        
    def work(self,):
        while True:
            logger.debug('worker #{} about to receive next task'.format(self.rank))
            task,ndata,opts = self.comm.recv(None,source=0)
            
            nproc = self.comm.Get_size()
            
            if task=='mvc':
                self.do_mvc(ndata, **opts)
            elif task=='psi':
                self.do_psi(ndata, **opts)
            elif task=='aci':
                self.do_aci(ndata, **opts)
            elif task=='duald':
                self.do_duald(ndata, **opts)
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
