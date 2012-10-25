import numpy as np
from numpy.ctypeslib import ndpointer
import os
import ctypes

 
CBType = ctypes.CFUNCTYPE(
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int)
 
def fastPD(
        unary,
        pairs, 
        binary, 
        wpairs=None,
        niters=100,
        debug=False,
        ):
    ''' MRF solve
    
        with N nodes, L labels, P pairs
    
    args:   
        unary: N x L array (unary potentials)
        pairs: P x 2 array (vertex indices for each pair)
        binary: L x L array (potential for each pair of labels)
        wpairs: P x 1 array (potential for each pair. All set to 1. if None)
        niters: (number of iterations. default is 100)
        
    returns:
        results: N x 1 array (computed label for each None)
        energy: (energy value of solution)
    
    '''
   
        
    unary_ = np.array(unary,dtype=np.float32)
    pairs_ = np.asarray(pairs,dtype=np.int32)
    binary_ = np.asarray(binary,dtype=np.float32)
    
    if wpairs is None:
        wedges_ = np.ones(len(pairs_)).astype(np.float32)
    else:
        wedges_ = wpairs.astype(np.float32)
    
    nnodes = unary_.shape[0]
    nlabels = unary_.shape[1]
    npairs = pairs_.shape[0]
    
    results = np.zeros(nnodes,dtype=np.int32)
    
    max_iter = niters
    
    energy = _libFastPD.fastPD(
        ctypes.c_int32(nnodes),
        ctypes.c_int32(nlabels),
        unary_.flatten(),
        ctypes.c_int32(npairs),
        pairs_.flatten(),
        binary_.flatten(),
        wedges_,
        results,
        ctypes.c_int32(max_iter),
        ctypes.c_bool(debug),
        )
    '''
    int nnodes,           // number of nodes
    int nlabels,          // number of labels
    T const* unary,       // unary potentials
    int npairs,           // number of pairs
    int const * pairs,    // pairs (ith pair = pairs[2*i], pairs[2*i+1])
    T const * binary,     // binary potentials
    T const * wedges,     // edge weighting
    int* results,         // return array
    int max_iter,         // max number of iterations
    bool debug            // print debug info
    '''
 
    del unary_, pairs_, binary_, wedges_
        
    return results, energy
      

      
def fastPD_potts(
        unary,
        labels,
        pairs, 
        wpairs,
        niters=100,
        debug=False,
        ):
    ''' MRF solve
    
        with N nodes, L labels, P pairs
    
    args:   
        unary: N x L array (unary potentials)
        labels: N x L array of labels
        pairs: P x 2 array (vertex indices for each pair)
        wpairs: P x 1 array (potential for each pair. All set to 1. if None)
        niters: (number of iterations. default is 100)
        
    returns:
        results: N x 1 array (computed label for each None)
        energy: (energy value of solution)
    
    '''
   
        
    unary_ = np.array(unary, dtype=np.float32)
    pairs_ = np.asarray(pairs, dtype=np.int32)
    labels_ = np.asarray(labels, dtype=np.int32)
    wedges_ = wpairs.astype(np.float32)
    
    nnodes = unary_.shape[0]
    nlabels = unary_.shape[1]
    npairs = pairs_.shape[0]
    
    results = np.zeros(nnodes,dtype=np.int32)
    _init = np.zeros(nnodes,dtype=np.int32,order='C')
    
    max_iter = niters
    
    energy = _libFastPD.fastPD_potts(
        ctypes.c_int32(nnodes),
        ctypes.c_int32(nlabels),
        unary_.flatten(),
        labels_.flatten(),
        ctypes.c_int32(npairs),
        pairs_.flatten(),
        wedges_,
        results,
        _init,
        ctypes.c_int32(max_iter),
        ctypes.c_bool(debug),
        )
    
    del unary_, pairs_, labels_, wedges_
        
    return results, energy
      
      

def fastPD_callback(
        unary,
        pairs,
        cost_function,
        niters=10000,
        debug=False,
        ):
    ''' MRF graph matching
    
    arguments:
        cf. fastPD()
        cost_function prototype:
            cost = cost_function(pair id, label 1, label 2)
    
    returns:
        results: N x 1 array (computed label for each None)
        energy: (energy value of solution)
    
    '''

    _unary = np.array(unary,dtype=np.float32,order='C')
    _pairs = np.asarray(pairs,dtype=np.int32,order='C')

    nnodes = _unary.shape[0]
    nlabels = _unary.shape[1]
    npairs = _pairs.shape[0]

    _results = np.zeros(nnodes,dtype=np.int32,order='C')
  
    _init = np.zeros(nnodes,dtype=np.int32,order='C')
    
    # cost_f = lambda a,b,c: np.float32(cost_function(a,b,c))
    
    energy = _libFastPD.fastPD_callback(
        ctypes.c_int32(nnodes),
        ctypes.c_int32(nlabels),
        _unary.flatten(),
        ctypes.c_int32(npairs),
        _pairs.flatten(),
        CBType(cost_function),
        _results,
        _init,
        ctypes.c_int32(niters),
        ctypes.c_bool(debug),
        )

    del _pairs,_unary, _init
    return _results, energy
    

    
    
def compute_energy_callback(solution, unary, pairs, cost_function, display=True):
    e_un = np.sum(unary[range(len(unary)), solution])    
    e_bi = 0
    for ip,p in enumerate(pairs):
        e_bi += cost_function(ip, solution[p[0]], solution[p[1]])
    
    if display:
        print 'energy unary = %f' %e_un    
        print 'energy binary = %f' %e_bi
        print 'total energy = %f' %(e_un + e_bi)
        
    return e_bi + e_un

    
#------------------------------------------------------------------------------

# import dlls
thispath = os.path.dirname(__file__)
if not len(thispath): thispath = '.'
thispath += '/'


#config='Release'
# config='Debug'
#flibFastPD = thispath + 'build/lib/%s/FastPD.dll' %config
flibFastPD = thispath + 'build/lib/libfastPD.so'

if os.path.isfile(flibFastPD):
    _libFastPD = ctypes.CDLL(flibFastPD)
    
    # arg types
    _libFastPD.fastPD.restype = ctypes.c_float
    _libFastPD.fastPD.argtypes = [
        ctypes.c_int32,                       #nnodes
        ctypes.c_int32,                       #nlabels
        ndpointer(dtype=np.float32,ndim=1),      #*unary
        ctypes.c_int32,                       #npairs
        ndpointer(dtype=np.int32,ndim=1),     #*pairs
        ndpointer(dtype=np.float32,ndim=1),      #*binary
        ndpointer(dtype=np.float32,ndim=1),      #*wedges
        ndpointer(dtype=np.int32,ndim=1),     #*results
        ctypes.c_int32,   #maxiter
        ctypes.c_bool,  #debug
        ]
        
    # arg types
    _libFastPD.fastPD_potts.restype = ctypes.c_float
    _libFastPD.fastPD_potts.argtypes = [
        ctypes.c_int32,                       #nnodes
        ctypes.c_int32,                       #nlabels
        ndpointer(dtype=np.float32,ndim=1),      #*unary
        ndpointer(dtype=np.int32,ndim=1),      #*labels
        ctypes.c_int32,                       #npairs
        ndpointer(dtype=np.int32,ndim=1),     #*pairs
        ndpointer(dtype=np.float32,ndim=1),      #*wedges
        ndpointer(dtype=np.int32,ndim=1),     #*results
        ndpointer(dtype=np.int32,ndim=1),     #*init
        ctypes.c_int32,   #maxiter
        ctypes.c_bool,  #debug
        ]
        
    _libFastPD.fastPD_callback.restype = ctypes.c_float
    _libFastPD.fastPD_callback.argtypes = [
        ctypes.c_int32,                       #nnodes
        ctypes.c_int32,                       #nlabels
        ndpointer(dtype=np.float32,ndim=1),      #*unary
        ctypes.c_int32,                       #npairs
        ndpointer(dtype=np.int32,ndim=1),     #*pairs
        CBType,                             # callback function
        ndpointer(dtype=np.int32,ndim=1),     #*results
        ndpointer(dtype=np.int32,ndim=1),     #*init
        ctypes.c_int32,   #maxiter
        ctypes.c_bool,  #debug
        ]

else:
    print 'could not find FastPD library:', flibFastPD
    
    
#------------------------------------------------------------------------------
if __name__=='__main__':
    print  'test_graph'
    nlabels = 2
    #unary = np.array([
    #    [10,1],
    #    [1,10],
    #    [10,1],
    #    ])
    
    #pairs = np.array([[0,1],[1,2],[2,0]])
    #binary = np.ones((nlabels,nlabels))
    #binary = np.array([
    #    [1, 50],
    #    [50, 1],
    #    ])
       
    N = 2000
    unary = np.random.random((N,10))*10
    pairs0 = np.random.randint(0,N-101,N*10)
    pairs1 = pairs0 + np.random.randint(0,100, N*10)
    pairs = np.c_[pairs0, pairs1]


    def costfunction(e,l1,l2): 
        # print e,l1,l2
        #return binary[l1,l2]
        cost =  1 + (l1!=l2)*np.random.random()
        #print cost
        return cost
      
    #res1 =  fastPD(unary, pairs, binary,debug=True)
    res2 =  fastPD_callback(unary, pairs, costfunction,debug=True)
    

    #print res1
    print res2
    
