import numpy as np
from scipy import ndimage


def sample_points(im, step,  mask=None):
 
    shape = np.asarray(im.shape, dtype=int)
    steps = np.ones(im.ndim, dtype=int)*step   

    ## make regular grid
    grid = np.zeros(shape)
    grid[::steps[0], ::steps[1], ::steps[2]] = 1

    if mask is not None:
        grid = grid * mask

    points0 = np.argwhere(grid)
    points = points0   
 
    ## move points
    emap = np.sqrt(np.sum([ndimage.sobel(im, axis=d)**2 for d in range(im.ndim)], axis=0))
    gradient = np.asarray(np.gradient(emap)).astype(float)
    for i in range(np.min(steps)/2):
        dp = gradient[(slice(None),) + tuple(points.T)].T
        dp = dp/np.c_[np.maximum(np.max(np.abs(dp), axis=1),1e-10)]
        dp = (dp + 0.5).astype(int)
        
        points = points - dp
        points = np.c_[
            np.clip(points[:,0],0, shape[0]-1),
            np.clip(points[:,1],0, shape[1]-1),
            np.clip(points[:,2],0, shape[2]-1),
            ]

    return points

from fast_marching import fastm
def get_edges(im, pts):
    
     speed = 1/np.sqrt(np.sum(
         [ndimage.sobel(im,axis=d)**2 for d in range(im.ndim)],axis=0) + 1e-5)
     
     ## compute edges
     labels, dist, edges, edgev = fastm.fast_marching_3d(
        speed,
        pts,
        heap_size=1e6,
        offset=1e-4,
        output_arguments=('labels', 'distances', 'edges', 'edge_values'),
        ) 
     
     return edges, edgev

def get_profiles(im, points, edges, rad=1):

    emap = np.sqrt(np.sum(
         [ndimage.sobel(im,axis=d)**2 for d in range(im.ndim)],axis=0) + 1e-5)
    emap /= np.std(emap)
 
    ## extract intensity
    dists = np.sqrt(np.sum((points[edges[:,0]] - points[edges[:,1]])**2,axis=1))
    profiles = []
    for i,e in enumerate(edges):
         pt0 = points[e[0]]
         pt1 = points[e[1]]
         
         vec   = (pt1 - pt0).astype(float)
         vec  /= np.max(np.abs(vec))
         if np.abs(vec[0])+np.abs(vec[1]) < 1e-5: 
             par1 = np.array([1.,0.,0.])
             par2 = np.array([0.,1.,0.])
         else:
             par1  = np.array([vec[1], -vec[0], 0])
             par1 /= np.sqrt(np.sum(par1**2))
             par2  = np.array([vec[0]*vec[2], vec[1]*vec[2], -(vec[0]**2 + vec[1]**2)])
             par2 /= np.sqrt(np.sum(par2**2))
         
         dist = dists[i]
         line = np.asarray([t*pt0 + (1-t)*pt1 for t in np.linspace(0,1,dist*2)])
         #line = (line + 0.5).astype(int)
         
         disk = (np.argwhere(np.ones((2*rad+1, 2*rad+1))) - rad)/float(rad)
         disk = disk[np.sum(disk**2,axis=1) < 1.00000001]
         profile = 0
         for d in disk:
             transl = par1*d[0] + par2*d[1]
             tline = (line + transl + 0.5).astype(int)
             if not np.all(np.all(tline>=[0,0,0],axis=1)&np.all(tline<im.shape,axis=1)):
                 continue
             profile = profile + emap[tuple(tline.T)]
         profile /= len(disk)           

         profiles.append(profile)
        # profiles[e[1]].append([e[0],profile])


    return profiles

def interpolate_profiles(profiles,size=None):
    from scipy import interpolate  
    if size is None:
        size = 0
        for profile in profiles:
            size += len(profile)/float(len(profiles))
    size = int(size + 1)
    x = []
    for profile in profiles:
        n = len(profile)
        if n<4:
            interpolator = interpolate.interp1d(np.linspace(0,1,n), profile, kind='linear')
        else:
            interpolator = interpolate.interp1d(np.linspace(0,1,n), profile, kind='cubic')
        d = interpolator(np.linspace(0,1,size))
        x.append(d.tolist() + [n])
    return x

def is_boundary(points, edges, seg):
    l1 = seg[tuple(points[edges[:,0]].T)]
    l2 = seg[tuple(points[edges[:,1]].T)]
    return l1==l2

def learn_profiles(x, z):
    import struct_svm 
    struct_svm.logger = struct_svm.utils_logging.get_logger('svm',struct_svm.utils_logging.INFO)
    from struct_svm import StructSVM
    
    S = [(x,z) for x,z in zip(x,z)]
    
    def loss(z,y,**kwargs):
        return 1*(y!=z) + 0.0
    
    def psi(x,y,**kwargs):
        if y==0: 
            return np.r_[x, [0.0 for i in x]]
        else:
            return np.r_[[0.0 for i in x],x]

    def mvc(w,x,z,**kwargs):
        scores = [np.dot(w,psi(x,y)) - loss(z,y) for y in [0,1]]
        #print z, scores
        return np.argmin(scores)

    svm = StructSVM(            
        S,
        loss,psi,mvc,
        C=10,
        )
    w,xi,info = svm.train()
    return w

def classify_profiles(x,w):
    def psi(x,y):
        if y==0:
            return np.r_[x, [0.0 for i in x]]
        else:
            return np.r_[[0.0 for i in x],x]
    sol = []
    for d in x:
        scores = [np.dot(w,psi(d,y)) for y in [0,1]]
        sol.append(np.argmin(scores))
    return sol


