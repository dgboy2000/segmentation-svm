import sys
import os
import numpy as np


'''
weight_functions = {
    # 'std_b10'     : lambda im: weight_std(im, beta=10),
    # 'std_b50'     : lambda im: weight_std(im, beta=50),
    # 'std_b100'    : lambda im: weight_std(im, beta=100),
    # 'inv_b100o1'    : lambda im: weight_inv(im, beta=100,offset=1),
    # 'pdiff_r1b50'   : lambda im: weight_patch_diff(im, r0=1, beta=50),
    'pdiff_r1b100'   : lambda im: weight_patch_diff(im, r0=1, beta=100),
    }
'''
import utils_logging
logger = utils_logging.get_logger('weigth_functions',utils_logging.DEBUG)

def weight_std(image, beta=1.0):
    ''' standard weight function '''
    im = np.asarray(image)
    
    ## affinity matrix sparse data
    data = np.exp( - beta * np.r_[tuple([
        np.square(
            im.take(np.arange(im.shape[d]-1), axis=d).ravel() - \
            im.take(np.arange(1,im.shape[d]), axis=d).ravel(),
            )
        for d in range(im.ndim)
        ])])
    
    ## affinity matrix sparse indices
    inds = np.arange(im.size).reshape(im.shape)
    ij = (
        np.r_[tuple([inds.take(np.arange(im.shape[d]-1), axis=d).ravel() \
            for d in range(im.ndim)])],
        np.r_[tuple([inds.take(np.arange(1,im.shape[d]), axis=d).ravel() \
            for d in range(im.ndim)])],
        )
    
    return ij, data
    
##------------------------------------------------------------------------------
def weight_inv(image, beta=1., offset=1.):
    ''' inverse weight function '''
    im = np.asarray(image)
    
    ## affinity matrix sparse data
    data = np.r_[tuple([
        np.abs(
            im.take(np.arange(im.shape[d]-1), axis=d).ravel() - \
            im.take(np.arange(1,im.shape[d]), axis=d).ravel(),
            )
        for d in range(im.ndim)
        ])]
    data = 1. / (offset + beta*data)
    logger.debug('(weight value spread: {:.2}-{:.2})'.format(
        np.min(data), np.max(data)))
    
    ## affinity matrix sparse indices
    inds = np.arange(im.size).reshape(im.shape)
    ij = (
        np.r_[tuple([inds.take(np.arange(im.shape[d]-1), axis=d).ravel() \
            for d in range(im.ndim)])],
        np.r_[tuple([inds.take(np.arange(1,im.shape[d]), axis=d).ravel() \
            for d in range(im.ndim)])],
        )
    
    return ij, data 
    
##------------------------------------------------------------------------------
def weight_patch_diff(
        image, r0=1, step=1, beta=1e0, r1=None, gw=True,
        ):
    ''' '''
    
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
