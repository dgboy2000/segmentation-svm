import os
import sys
import numpy as np


         

# ------------------------------------------------------------------------------
def fast_marching_3d(
        image, 
        points,
        plabels=None,
        mask=None,
        heap_size=1e6,
        offset=0.0,
        connectivity26=False,
        output_arguments=('distances', 'labels'),
        method='standard',
        ):
        
        ''' return arguments '''
        output_names = ['distances', 'labels', 'parents', 'intersections' , 
                        'edges', 'edge_values', 'edge_size']
        if output_arguments not in output_names:
            for o in output_arguments:
                if o not in output_names:
                    print 'error, invalid argument: %s', o
                    sys.exit(1)
        
        '''method '''
        method_dict = {
            "standard": 0,
            "anisotropic": 1,
            "test": 2,
            }
        
        
        ''' experimental fast marching method '''
        im = np.asarray(image)
        cim = np.asarray(np.atleast_3d(im), order='C', dtype=np.float32)
        cpts = np.asarray(np.atleast_2d(points), order='C', dtype=np.uint)
        if plabels is None:
            cplab = np.arange(len(cpts), dtype=np.uint)
        else:
            cplab = np.asarray(plabels, dtype=np.uint)
        
        if mask is None:
            cmask = np.ones(cim.shape, dtype=np.bool8)
        else:
            cmask = np.asarray(mask, order='C',dtype=np.bool8)
        
        if cpts.shape[1] == 3:
            pass
        elif cpts.shape[1] == 2:
            cpts = np.asarray(
                np.c_[cpts, np.zeros(len(cpts), dtype=int)],
                order='C', dtype=np.uint)
        else:
            print "error in shape of point array"
            sys.exit(1)
        
        alloc = Allocator()
        
        lib.fast_marching_3d_general(
            ctypes.ARRAY(ctypes.c_uint,3)(*cim.shape),
            cim,
            len(cpts),
            cpts,
            alloc.cfunc,
            cplab,
            cmask,
            int(heap_size),
            offset,
            connectivity26,
            method_dict[method],
            False,
            )

        allocated_arrays = alloc.allocated_arrays
        output_dict = {
            'distances': allocated_arrays["distances"].reshape(im.shape),
            'labels': allocated_arrays["labels"].reshape(im.shape),
            'parents': allocated_arrays["parents"].reshape(im.shape),
            'intersections': allocated_arrays["intersections"].reshape(im.shape),
            'edges': allocated_arrays["edges"],
            'edge_values': allocated_arrays["edge_values"],
            'edge_size': allocated_arrays["edge_size"],
            }
            
        if output_arguments in output_dict.keys():
            return output_dict[output_arguments]
            
        output = []
        for arg in output_arguments:
            output.append(output_dict[arg])
        
        return tuple(output)
          

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
        
        
# ------------------------------------------------------------------------------
## load lib on import
import os
import platform
import ctypes
from numpy.ctypeslib import ndpointer

if platform.architecture()[0]=='64bit': build_dir = 'build64'
else: build_dir = 'build32'

config = 'Release'
# config = 'Debug'
libpath = 'fastm/%s/%s/fastm.dll' %(build_dir, config)
path = os.path.abspath(os.path.dirname(__file__))
if not len(path):
    path = './'
file = path + '//' + libpath
print file
if os.path.isfile(file):
    libffd = ctypes.CDLL(file)

    # arg types for fast_marching_3d_aniso
    lib.fast_marching_3d_general.argtypes = [
        ctypes.ARRAY(ctypes.c_uint,3),                           # img size
        ndpointer(dtype=np.float32,ndim=3,flags='C_CONTIGUOUS'), # image
        ctypes.c_uint,                                           # number of points
        ndpointer(dtype=np.uint,ndim=2,flags='C_CONTIGUOUS'),    # points
        Allocator.CFUNCTYPE,
        ndpointer(dtype=np.uint, ndim=1),                        # point labels
        ndpointer(dtype=np.bool8, ndim=3),                       # mask
        ctypes.c_uint,  # heap size
        ctypes.c_float, # gap offset
        ctypes.c_bool,  # connectivity = 26
        ctypes.c_int,   # method
        ctypes.c_bool,  # debug
        ]
    
    
# ------------------------------------------------------------------------------
if __name__=='__main__':
    from scipy.misc import imread

    # ''' 
    vol = imread('test/thigh.png', flatten=True)
    step = 10
    points = np.indices(vol.shape)[:,::step,::step].ravel().reshape((2,-1)).T
    

    labels, dist, edges, edgev = fast_marching_3d_aniso(
        vol,
        points,
        heap_size=1e5,
        offset=1e-5,
        output_arguments=('labels', 'distances', 'edges', 'edge_values'),
        )

    

        