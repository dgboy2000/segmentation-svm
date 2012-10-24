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
        cpts = np.asarray(np.atleast_2d(points), order='C', dtype=np.uint32)
        if plabels is None:
            cplab = np.arange(len(cpts), dtype=np.uint32)
        else:
            cplab = np.asarray(plabels, dtype=np.uint32)
        
        if mask is None:
            cmask = np.ones(cim.shape, dtype=np.bool8)
        else:
            cmask = np.asarray(mask, order='C',dtype=np.bool8)
        
        if cpts.shape[1] == 3:
            pass
        elif cpts.shape[1] == 2:
            cpts = np.asarray(
                np.c_[cpts, np.zeros(len(cpts), dtype=np.int32)],
                order='C', dtype=np.uint32)
        else:
            print "error in shape of point array"
            sys.exit(1)
        
        alloc = Allocator()
        
        lib.fast_marching_3d_general(
            ctypes.ARRAY(ctypes.c_uint32,3)(*cim.shape),
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
import ctypes
class Allocator:
    CFUNCTYPE = ctypes.CFUNCTYPE(
        ctypes.c_long, 
        ctypes.c_int32, 
        ctypes.POINTER(ctypes.c_uint32), 
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
import sys
import platform
from numpy.ctypeslib import ndpointer

if sys.platform[:3]=='win':
    libpath = 'build/fastm/libs/Release/fastm.dll'
else:
    libpath = 'build/fastm/libs/libfastm.so'
path = os.path.abspath(os.path.dirname(__file__))
if not len(path):
    path = './'
file = path + '//' + libpath
print file
if os.path.isfile(file):
    lib = ctypes.CDLL(file)

    # arg types for fast_marching_3d_aniso
    lib.fast_marching_3d_general.argtypes = [
        ctypes.ARRAY(ctypes.c_uint32,3),                           # img size
        ndpointer(dtype=np.float32,ndim=3,flags='C_CONTIGUOUS'), # image
        ctypes.c_uint32,                                           # number of points
        ndpointer(dtype=np.uint32,ndim=2,flags='C_CONTIGUOUS'),    # points
        Allocator.CFUNCTYPE,
        ndpointer(dtype=np.uint32, ndim=1),                        # point labels
        ndpointer(dtype=np.bool8, ndim=3),                       # mask
        ctypes.c_uint32,  # heap size
        ctypes.c_float, # gap offset
        ctypes.c_bool,  # connectivity = 26
        ctypes.c_int32,   # method
        ctypes.c_bool,  # debug
        ]
    
    
# ------------------------------------------------------------------------------
if __name__=='__main__':
    from scipy.misc import imread, imsave

    # ''' 
    vol = imread('test/thigh.png', flatten=True)
    step = 10
    #points = np.indices(vol.shape)[:,::step,::step].ravel().reshape((2,-1)).T
    grid = np.zeros(vol.shape)
    grid[::step, ::step] = 1
    points = np.argwhere(grid)

    speed = 1./np.sqrt(np.sum([g**2 for g in np.gradient(vol)], axis=0))
    labels, dist, edges, edgev = fast_marching_3d(
        speed,
        points,
        heap_size=1e6,
        offset=1e-2,
        output_arguments=('labels', 'distances', 'edges', 'edge_values'),
        )
    imsave('test/labels.png', labels)
    imsave('test/dist.png', (dist*1000/np.max(dist)).astype(int))

    

        
