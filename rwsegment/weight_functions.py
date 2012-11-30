import sys
import os
import numpy as np


import utils_logging
logger = utils_logging.get_logger('weigth_functions',utils_logging.INFO)


def weight_std(image, i, j, beta=1.0, omega=0):
    ''' standard weight function 
    
        for touching pixel pair (i,j),
            wij = exp (- beta (image.flat[i] - image.flat[j])^2)
    '''
    im = np.asarray(image)
    wij = np.exp(-beta * (image.flat[i] - image.flat[j])**2) + omega
    return wij

##------------------------------------------------------------------------------
def weight_inv(image, i, j, beta=1., offset=1.):
    ''' inverse weight function '''
    im = np.asarray(image)
    data = np.abs(image.flat[i] - image.flat[j])
    wij = 1. / (offset + beta*data)
    return wij
    
##------------------------------------------------------------------------------
def weight_patch_diff(
        image, r0=1, step=1, beta=1e0, r1=None, gw=True,
        ):
    print "not implemented"
    sys.exit(1)

    '''
    im = np.asarray(image)
    
    ## make 3d arrays
    if im.ndim < 3:
        im = np.atleast_3d(im)
    elif im.ndim==3:
        pass
    else:
        return 0
    im = np.require(im, dtype=np.float32, requirements=['C','O'])
    
    shape = ctypes.ARRAY(ctypes.c_uint,3)(*im.shape)
    
    if r1 is None: r1 = r0
    rad = ctypes.ARRAY(ctypes.c_uint,2)(r0,r1)
    
    ## run ctypes lib
    alloc = Allocator()
    
    try:
        clib.patch_diff_3d(
            shape,
            im,
            rad,
            step,
            alloc.cfunc,
            gw,
            )
    except:
        print 'Warning: ctypes library, requires compilation'
        sys.exit(1)
    allocated_arrays = alloc.allocated_arrays
    
    ## get output data
    ij   = tuple(allocated_arrays['ij'].T)
    diff = allocated_arrays['data']
    data = np.exp(-beta*diff)
    
    return ij, data
    '''
    
    
##------------------------------------------------------------------------------
import os
import platform
import ctypes
from numpy.ctypeslib import ndpointer

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



''' load library '''
if sys.platform[:3]=='win':
    if platform.architecture()[0]=='64bit': build_dir = 'build64'
    else: build_dir = 'build32'
    
    config = 'Release'
    # config = 'Debug'
    libpath = 'lfunctions/%s/%s/lfunctions.dll' %(build_dir, config)
else:
    libpath = 'lfunctions/build/libs/liblfunctions.so'
path = os.path.abspath(os.path.dirname(__file__))
if not len(path):
    path = './'
file = path + '//' + libpath
# print file
if os.path.isfile(file):
    clib = ctypes.CDLL(file)

    # arg types
    clib.patch_diff_3d.argtypes = [
        ctypes.ARRAY(ctypes.c_uint,3),  # shape
        ndpointer(dtype=np.float32,ndim=3,flags='C_CONTIGUOUS'), # image
        ctypes.ARRAY(ctypes.c_uint,2),  # radius
        ctypes.c_uint,                  # step
        Allocator.CFUNCTYPE,
        ctypes.c_bool,                  # gaussian_weighting
        ]
