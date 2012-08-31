try:
    from mpi4py import MPI
    RANK = MPI.COMM_WORLD.Get_rank()
    SIZE = MPI.COMM_WORLD.Get_size()
    COMM = MPI.COMM_WORLD
except ImportError:
    RANK = 0
    SIZE = 1
    COMM = None