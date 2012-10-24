import numpy as np
from numpy.ctypeslib import ndpointer
import os
import ctypes


    
def TRW_binary(
        unary, 
        pairs, 
        wpairs, 
        niters=1000, 
        randomize=False,
        use_bp=False,
        verbose=False,        
        ):
    
    
    _unary = np.ascontiguousarray(unary,dtype=float)
    nnodes = _unary.shape[0]

    
    _pairs = np.ascontiguousarray(pairs,dtype=int)
    _wpairs = np.ascontiguousarray(wpairs,dtype=float)
    npairs = _pairs.shape[0]
    
    if _unary.shape[1]!=2 or _wpairs.shape[1]!=4: 
        print 'error, this function takes only binary labels'
        return
    
    solution = np.zeros(nnodes,dtype=int,order='C')

    energy = _libTRW.trw_binary(
        nnodes, _unary,
        npairs, _pairs, _wpairs,
        solution,
        niters, randomize, use_bp, verbose,
        )
 
    del _pairs,_unary, _wpairs
    
    return solution, energy

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def TRW_general(
        unary, 
        pairs, 
        wpairs, 
        niters=1000,
        randomize=False,
        use_bp=False,
        verbose=False,
        output_arguments=('solution', 'energy'),
        ):
    '''
    unary -> N x L
    pairs -> P x 2
    wpairs -> P x L^2, wpairs[i,j] = wi(l1 = j%L, l2 = j/L)
    
    '''
    
    _unary = np.ascontiguousarray(unary,dtype=float)
    nnodes = _unary.shape[0]
    nlabels = _unary.shape[1]
    
    _pairs = np.ascontiguousarray(pairs,dtype=int)
    _wpairs = np.ascontiguousarray(wpairs,dtype=float)
    npairs = _pairs.shape[0]
    
    if _wpairs.shape[1]!=nlabels**2: 
        print 'error in shape of binary costs'
        return
    
    solution = np.zeros(nnodes,dtype=int,order='C')

    if 'min_marginals' in output_arguments:
        compute_min_marginals=True
    else: compute_min_marginals=False
    
    alloc = Allocator()
    energy = _libTRW.trw_general(
        nnodes, nlabels, _unary,
        npairs, _pairs, _wpairs,
        #solution,
        alloc.cfunc,
        int(niters),
        compute_min_marginals,
        randomize, 
        use_bp,
        verbose,
        ) 
        
    allocated_arrays = alloc.allocated_arrays
    output = []
    for arg in output_arguments:
        if arg in allocated_arrays:
            output.append(allocated_arrays[arg])
        elif arg=='energy':
            output.append(energy)
            
    # del _pairs,_unary, _wpairs
    if len(output)==1: return output[0]
    else: return tuple(output)
    # return solution, energy
    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
 
def TRW_potts(
        unary, 
        pairs, 
        wpairs, 
        labels=None,
        niters=1000, 
        randomize=False,
        use_bp=False,
        verbose=False,        
        ):
    '''
    unary -> N x L
    pairs -> P x 2
    wpairs -> P x 1
    
    '''
    
    _unary = np.ascontiguousarray(unary, dtype=float)
    nnodes = _unary.shape[0]
    nlabels = _unary.shape[1]
    if not labels is None:
        _labels = np.ascontiguousarray(labels, dtype=int)
    
    _pairs = np.ascontiguousarray(pairs,dtype=int)
    _wpairs = np.ascontiguousarray(wpairs,dtype=float)
    npairs = _pairs.shape[0]
    
    if _wpairs.size != npairs: 
        print 'error in shape of binary costs'
        return
    
    solution = np.zeros(nnodes,dtype=int,order='C')

    if labels is None:
        energy = _libTRW.trw_potts(
            nnodes, nlabels, _unary,
            npairs, _pairs, _wpairs,
            solution,
            int(niters),
            randomize, 
            use_bp, 
            verbose,
            ) 
    else:
        energy = _libTRW.trw_potts_wlabels(
            nnodes, nlabels, _unary, _labels,
            npairs, _pairs, _wpairs,
            solution,
            int(niters),
            randomize, 
            use_bp, 
            verbose,
            ) 
    del _pairs, _unary, _wpairs
    if not labels is None: del _labels
    
    return solution, energy
    
    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def compute_energy(solution, unary, pairs, wpairs, display=True):
    e_un = np.sum(unary[range(len(unary)), solution])    
    e_bi = 0
    nlabels = unary.shape[1]
    for ip,p in enumerate(pairs):
        e_bi += wpairs[ip, solution[p[0]] + nlabels*solution[p[1]]]
    
    if display:
        print 'energy unary = %f' %e_un    
        print 'energy binary = %f' %e_bi
        print 'total energy = %f' %(e_un + e_bi)
        
    return e_un, e_bi
    

    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class Allocator:
    CFUNCTYPE = ctypes.CFUNCTYPE(
        ctypes.c_long, 
        ctypes.c_int, 
        ctypes.POINTER(ctypes.c_uint), 
        ctypes.c_char_p,
        ctypes.c_char_p,
        )
    
    def __init__(self):
        self.allocated_arrays = {}
    
    def __call__(self, dims, shape, dtype, name):
        arr = np.empty(shape[:dims], np.dtype(dtype))
        self.allocated_arrays[name] = arr
        return arr.ctypes.data_as(ctypes.c_void_p).value

    def getcfunc(self):
        return self.CFUNCTYPE(self)
    cfunc = property(getcfunc)
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# import dlls
thispath = os.path.dirname(__file__)
if not len(thispath): thispath = '.'
thispath += '/'

#config='Release'
# config='Debug'

#flibTRW = thispath + 'build/lib/%s/trw.dll' %config
flibTRW = thispath + 'build/lib/libtrw.so'

if os.path.isfile(flibTRW):

    _libTRW = ctypes.CDLL(flibTRW)

    # arg types
    _libTRW.trw_binary.restype = ctypes.c_double
    _libTRW.trw_binary.argtypes = [
        ctypes.c_int,                       #nnodes
        ndpointer(dtype=float,
                  ndim=2, flags='C'),       #*unary
        ctypes.c_int,                       #npairs
        ndpointer(dtype=int,
                  ndim=2,flags='C'),        #*pairs
        ndpointer(dtype=float,
                  ndim=2,flags='C'),        #*wpairs
        ndpointer(dtype=int,
                  ndim=1,flags='C'),        #*solution
        ctypes.c_int,                       #maxiter
        ctypes.c_bool,                      #randomize order
        ctypes.c_bool,                      #use belief propagation algo
        ctypes.c_bool,                      #debug
        ]
        
    # arg types
    _libTRW.trw_general.restype = ctypes.c_double
    _libTRW.trw_general.argtypes = [
        ctypes.c_int,                       #nnodes
        ctypes.c_int,                       #nlabels
        ndpointer(dtype=float,
                  ndim=2, flags='C'),       #*unary
        ctypes.c_int,                       #npairs
        ndpointer(dtype=int,
                  ndim=2,flags='C'),        #*pairs
        ndpointer(dtype=float,
                  ndim=2,flags='C'),        #*bpots
        # ndpointer(dtype=int,
                  # ndim=1,flags='C'),        #*solution
        Allocator.CFUNCTYPE,
        ctypes.c_int,                       #maxiter
        ctypes.c_bool,                      #compute_min_marginals
        ctypes.c_bool,                      #randomize order
        ctypes.c_bool,                      #use belief propagation algo
        ctypes.c_bool,                      #debug
        ]
        
    # arg types
    _libTRW.trw_potts.restype = ctypes.c_double
    _libTRW.trw_potts.argtypes = [
        ctypes.c_int,                       #nnodes
        ctypes.c_int,                       #nlabels
        ndpointer(dtype=float,
                  ndim=2, flags='C'),       #*unary
        ctypes.c_int,                       #npairs
        ndpointer(dtype=int,
                  ndim=2,flags='C'),        #*pairs
        ndpointer(dtype=float,
                  ndim=1,flags='C'),        #*wpairs
        ndpointer(dtype=int,
                  ndim=1,flags='C'),        #*solution
        ctypes.c_int,                       #maxiter
        ctypes.c_bool,                      #randomize order
        ctypes.c_bool,                      #use belief propagation algo
        ctypes.c_bool,                      #debug
        ]
        
    # arg types
    _libTRW.trw_potts_wlabels.restype = ctypes.c_double
    _libTRW.trw_potts_wlabels.argtypes = [
        ctypes.c_int,                       #nnodes
        ctypes.c_int,                       #nlabels max
        ndpointer(dtype=float,
                  ndim=2, flags='C'),       #*unary
        ndpointer(dtype=int,                  
                  ndim=2, flags='C'),       #*labels
        ctypes.c_int,                       #npairs
        ndpointer(dtype=int,
                  ndim=2,flags='C'),        #*pairs
        ndpointer(dtype=float,
                  ndim=1,flags='C'),        #*wpairs
        ndpointer(dtype=int,
                  ndim=1,flags='C'),        #*solution
        ctypes.c_int,                       #maxiter
        ctypes.c_bool,                      #randomize order
        ctypes.c_bool,                      #use belief propagation algo
        ctypes.c_bool,                      #debug
        ]
        
else:
    print 'could not find trw library:', flibTRW
    

#------------------------------------------------------------------------------
if __name__=='__main__':
    print 'test binary MRF'
    unary = [[0, 1],[100, 1],[0, 100]]
    pairs = [[0,1],[1,2],[2,0]]
    wpairs = [[0,10,10,0],[0,10,10,0],[0,10,10,0]]
    sol, en = TRW_binary(unary,pairs,wpairs)
    print sol
    print en
    
    
