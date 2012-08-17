

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def process(d):
    return d*100

if rank == 0:
   data = [np.asarray((i+1)**2) for i in range(size)]
else:
   data = None


data = comm.scatter(data,root=0)

res = process(data)

res = comm.gather(res,root=0)

if rank==0:
    print 'data=',res
else:
    assert data is None

