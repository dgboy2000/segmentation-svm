import numpy as np
from scipy import ndimage


def sample_points(im, step,  mask=None, maxiter=20):
 
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
    fim = ndimage.gaussian_filter(im.astype(float), sigma=1.0)
    emap = np.sqrt(np.sum([ndimage.sobel(fim, axis=d)**2 for d in range(im.ndim)], axis=0))
    gradient = np.asarray(np.gradient(emap)).astype(float)
    for i in range(np.max(steps)/2):
	axes = i < (steps/2)
        if i >= maxiter: break
        dp = gradient[(slice(None),) + tuple(points.T)].T
        dp = dp/np.c_[np.maximum(np.max(np.abs(dp), axis=1),1e-10)]
        dp = (dp + 0.5*(-1)**(dp<0)).astype(int)
        
        for axe in np.where(axes)[0]:
            points[:,axe] = (points - dp)[:,axe]
        points = points[np.all(points>0, axis=1)&np.all(points<shape,axis=1)]

    return points

from fast_marching import fastm
def get_edges(im, pts, mask=None):
    
     speed = 1/np.sqrt(np.sum(
         [ndimage.sobel(im,axis=d)**2 for d in range(im.ndim)],axis=0) + 1e-5)
     
     ## compute edges
     labels, dist, edges, edgev = fastm.fast_marching_3d(
        speed,
        pts,
        heap_size=1e6,
        offset=1e-2,
        mask=mask,
        output_arguments=('labels', 'distances', 'edges', 'edge_values'),
        ) 
     
     return edges, edgev, labels

def get_profiles(im, points, edges, rad=0):

    emap = np.sqrt(np.sum(
         [ndimage.sobel(im,axis=d)**2 for d in range(im.ndim)],axis=0) + 1e-5)
    emap = emap - np.min(emap)
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
         
         dist = int(dists[i] + 1)
         line = np.asarray([(1-t)*pt0 + t*pt1 for t in np.linspace(0,1,dist)])
         
         disk = (np.argwhere(np.ones((2*rad+1, 2*rad+1))) - rad)/float(rad + 1*(rad==0))
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


    return profiles, emap, dists

def make_features(profiles,size=None, additional=None):
    from scipy import interpolate  
    if size is None:
        size = 0
        for profile in profiles:
            size += len(profile)/float(len(profiles))
        size = int(size + 1)
    sizes = [size/4, size/2, size]
    x = []
    for i,profile in enumerate(profiles):
        n = len(profile)
        feature = []
        if n<4:
            interpolator = interpolate.interp1d(np.linspace(0,1,n), profile, kind='linear')
        else:
            interpolator = interpolate.interp1d(np.linspace(0,1,n), profile, kind='cubic')
        for size in sizes:
            d = interpolator(np.linspace(0,1,size))
            feature.extend(d.tolist())
        ## add features, including average, std, min, max, length
        x.append(feature + [np.mean(d), np.std(d), np.max(d), np.min(d),n])
        if additional is not None:
            x[-1].extend([ad[i] for ad in additional])
    return x

def is_boundary(points, edges, seg):
    l1 = seg[tuple(points[edges[:,0]].T)]
    l2 = seg[tuple(points[edges[:,1]].T)]
    return l1!=l2

class Classifier(object):
    def __init__(self,w=None):
        if w is not None:
            self.w = w

    def loss(self,z,y,**kwargs):
        return 1*(y!=z) + 0.0
    
    def psi(self,x,y,**kwargs):
        if y==0: 
            return np.r_[x, [0.0 for i in x]]
        else:
            return np.r_[[0.0 for i in x],x]

    def mvc(self,w,x,z,**kwargs):
        scores = [np.dot(w,self.psi(x,y)) - self.loss(z,y) for y in [0,1]]
        #print z, scores
        return np.argmin(scores)


    def train(self, x, z, balanced=True, **kwargs):
        import struct_svm 
        struct_svm.logger = struct_svm.utils_logging.get_logger('svm',struct_svm.utils_logging.INFO)
        from struct_svm import StructSVM
        
        C = kwargs.pop('C', 1.)

        if balanced:
            n0 = np.where(np.asarray(z)==0)[0]
            n1 = np.where(np.asarray(z)==1)[0]
            n = np.minimum(len(n0), len(n1))
            in0 = np.random.permutation(n0)[:n]
            in1 = np.random.permutation(n1)[:n]
            S  = [(x[i],z[i]) for i in in0]
            S += [(x[i],z[i]) for i in in1]
        else:   
            S = [(x[i],z[i]) for i in range(len(z))]
        
        svm = StructSVM(S, self.loss, self.psi, self.mvc,C=C,)
         
        ## train svm        
        w,xi,info = svm.train()
        self.w = w
        self.xi = xi

    def classify(self,x):
        w = self.w
        sol = []
        scores = []
        for d in x:
            score = [np.dot(w,self.psi(d,y)) for y in [0,1]]
            y = np.argmin(score)
            sol.append(y)
            scores.append(score)
        return sol, scores
    

