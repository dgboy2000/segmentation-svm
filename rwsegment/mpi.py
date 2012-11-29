
import sys
if '--parallel' in sys.argv:
    from mpi4py import MPI
    RANK = MPI.COMM_WORLD.Get_rank()
    SIZE = MPI.COMM_WORLD.Get_size()
    COMM = MPI.COMM_WORLD
    HOST = MPI.Get_processor_name()
else:
    import platform
    RANK = 0
    SIZE = 1
    COMM = None
    HOST = platform.processor()


