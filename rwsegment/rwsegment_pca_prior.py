import sys
import os
import numpy as np


from scipy import sparse
from scipy.sparse import linalg as splinalg


## compute prior 
class PriorGenerator:
    ''' generate prior
    '''
    
    def __init__(self, labelset, mask=None):
        self.x0     = 0
        self.ntrain = 0
        #self.mask   = 0
        self.vec    = []
        
        self.mask = mask
        
        ## intensity prior
        self.im_ntrain = None
        self.labelset = np.asarray(labelset)
        
        
    def add_training_data(self, atlas, image=None, nrandom=10):
        seg = np.asarray(atlas, dtype=int)
       
        ## set unwanted labels to background label (labelset[0])
        bg = self.labelset[0]
        seg.flat[~np.in1d(seg, self.labelset)] = bg
        
        ## compute background mask
        #self.mask = self.mask | (seg!=bg)
       
        if nrandom>0:
            print 'add {} random segmentations'.format(nrandom)
            segs = compute_random_segs(seg, nrandom=nrandom)
            segs = [seg] + segs
        else:
            segs = [seg]

        for a in segs:
            if self.mask is not None:
                b = a[self.mask]
            else: b = a.ravel()
 
            ## binary assignment matrix
            bin = b==np.c_[self.labelset]
            
            ## compute average
            x = bin.astype(float)
            self.x0     = self.x0 + x
            self.ntrain += 1
            
            ## compute covariance
            self.vec.append(x)

       ## if im is provided, compute average and std of intensity
        if image is not None:
            nim = image/np.std(image)
            if self.im_ntrain is None:
                self.im_avg    = np.zeros(len(self.labelset))
                self.im_avg2   = np.zeros(len(self.labelset))
                self.im_ntrain = np.zeros(len(self.labelset), dtype=int)
               
            for label in range(len(self.labelset)):
                inds = np.where(bin[label])[0]
                self.im_avg[label]    += np.sum(nim.flat[inds])
                self.im_avg2[label]   += np.sum(nim.flat[inds]**2)
                self.im_ntrain[label] += len(inds)
                
        
    def get_mask(self):
        return self.mask
        
    def get_prior(self,*args,**kwargs):

        neigval = 15

        nlabel = len(self.labelset)
        if self.mask is not None:
            imask = np.where(self.mask.ravel())[0]
        else:
            imask = np.arange(len(self.x0[0]))

        ## average probability
        #mean = self.x0[:,imask] / float(self.ntrain)
        mean = self.x0 / float(self.ntrain)
        
        ## variance
        # x is binary, thus we have avg(x) = avg(x**2)
        var = mean  - mean**2
         
        ## covariance
        #vecs = np.asarray([v[:,imask].ravel() for v in self.vec]).T - np.c_[mean.ravel()]
        vecs = np.asarray([v.ravel() for v in self.vec]).T - np.c_[mean.ravel()]

        nvec = len(self.vec)
        neig = np.minimum(neigval, nvec-1)
        nvar = self.vec[0].size
        cov = np.zeros((nvec,nvec))
        for i1, vec1 in enumerate(vecs.T):
            for i2, vec2 in enumerate(vecs.T[:i1+1]):
                cov[i1,i2] += 1/float(nvec)*np.dot(vec1, np.c_[vec2])
        cov = cov + np.triu(cov.T,k=1)
        w,v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w = w[order[:neig]]
        v = v[:,order[:neig]]
        U = np.dot(vecs, v)
        U = U/np.sqrt(np.sum(U**2,axis=0))

        ## prior dict
        prior = {
            'labelset': self.labelset,
            'imask': imask, 
            'data': mean, 
            'variance': var,
            'eigenvectors': U,
            }
        
        ## if intensity prior
        if self.im_ntrain is not None:
            im_avg = self.im_avg / self.im_ntrain.astype(float)
            im_var = self.im_avg2 / self.im_ntrain.astype(float) - im_avg**2
            
            ## add to prior dict
            prior['intensity']= (im_avg, im_var)
        
        return prior
       
 
def compute_random_segs(seg, nrandom=1, step=(10,30,30)):
    from ffd import ffd

    steps = step*np.ones(3,dtype=int)
    gshape = seg.shape/steps

    gnorm = np.asarray(gshape) - 1.0
    grid = np.argwhere(np.ones(gshape))/gnorm * seg.shape
     
    dffd_ = ffd.FFD((0,0,0) + seg.shape, shape=5,degree=3)
    dffd = ffd.EFFD(dffd_.controls, dffd_.cells)
    
    impoints = np.argwhere(np.ones(seg.shape))
    dffd.setpoints(impoints, use_clib=True)

    controls = np.array(dffd.controls).astype(int)
    inner = np.all((controls!=[0,0,0])&(controls!=seg.shape),axis=1)
        

    bboxmin = np.min(np.argwhere(seg>0),axis=0)-steps
    bboxmax = np.max(np.argwhere(seg>0),axis=0)+steps
    inner &= np.all((controls>=bboxmin)&(controls<=bboxmax),axis=1)
    inner = np.where(inner)[0]

    print 'start randomize images'
    imgs = []
    for i in range(nrandom):
        rvals = np.c_[
            np.random.randint(-steps[0]*0.5,steps[0]*0.5,len(inner)),
            np.random.randint(-steps[1]*0.5,steps[1]*0.5,len(inner)),
            np.random.randint(-steps[2]*0.5,steps[2]*0.5,len(inner)),
            ]

        controls = np.array(dffd.controls).astype(int)
        controls[inner] += rvals

        dpts0 = dffd.deform(controls, use_clib=True)

        dpts = np.clip(dpts0,0,np.asarray(seg.shape)-1)
        dpts = (dpts + 0.5).astype(int)

        dseg = np.zeros(seg.shape, dtype=int)
        dseg.flat = seg[tuple(dpts.T)]
        imgs.append(dseg)
        
    return imgs
    
