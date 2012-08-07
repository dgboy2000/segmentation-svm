'''
    Notes:
        - the energy should only be computed wrt to the unkwnown pixels.
        In some place we sloppily include known pixels in the computation.
        These do not matter since a constant does not change the solution.
        
        - Change the smv training set to the non-registered image set.
        
        - idea: make one big call containing psi, loss and mvc 
        so that we can cache Laplacians (used in both psi and mvc)?
        (e.g. We could call that class UserFunctionsSVM)
        
        - the loss function (1-zy) is ambiguous: what is worse 0 or 1-z ?
        maybe (1-2z)y  is better ?
'''


import sys
import os
import logging

import numpy as np

##temp: load rwsegment lib
sys.path += [os.path.abspath('../')]

from rwsegment import ioanalyze
from rwsegment import weight_functions as wflib
from rwsegment import rwmean_svm
reload(rwmean_svm), reload(wflib)

import structsvmpy 
reload(structsvmpy)

def main():
    ## load volume names 
    import config
    reload(config)
    
    ## paths
    dir_reg     = config.dir_reg
    
    ## params
    # slices = [slice(20,40),slice(None),slice(None)]
    slices = [slice(None),slice(None),slice(None)]
    
    labelset = [0,13,14,15,16]
    
    ## rw params
    rwparams = {
        'labelset':labelset,
        'tol': 1e-3,
        'maxiter': 1e2,
        }
        
    ## svm params
    svmparams = {
        'C': 100,
        'nitermax':100,
        'loglevel':logging.INFO,
        }
        
    ## weight functions
    weight_functions = {
        'std_b10'     : lambda im: wflib.weight_std(im, beta=10),
        # 'std_b50'     : lambda im: wflib.weight_std(im, beta=50),
        # 'std_b100'    : lambda im: wflib.weight_std(im, beta=100),
        # 'inv_b100o1'  : lambda im: wflib.weight_inv(im, beta=100, offset=1),
        'pdiff_r1b50' : lambda im: wflib.weight_patch_diff(im, r0=1, beta=50),
        # 'pdiff_r1b100': lambda im: wflib.weight_patch_diff(im, r0=1, beta=100),
        }
        
        
    def process(test):
        ## learn rwmean parameters
        
        outdir = test
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        ## load mask and prior
        prior = None
        file_mask  = outdir + 'mask.hdr'
        file_prior = outdir + 'prior.npz'
        if os.path.exists(file_prior):
            mask  = ioanalyze.load(file_mask)
            prior = np.load(file_prior)
        else:
            generator = rwmean_svm.PriorGenerator(labelset)
            
        ## training images and segmentations
        training_set = []
        for train in config.vols:
            if test==train: continue
            print '  load training data: {}'.format(train)
            file_seg = dir_reg + test + train + 'regseg.hdr'
            file_im  = dir_reg + test + train + 'reggray.hdr'
            im  = ioanalyze.load(file_im)
            seg = ioanalyze.load(file_seg)
            
            ## make bin vector z from segmentation
            seg.flat[~np.in1d(seg.ravel(),labelset)] = labelset[0]
            bin = (np.c_[seg.ravel()]==labelset).ravel('F')
            
            ## normalize image by variance
            im = im/np.std(im)
            
            training_set.append((im, bin))
            
            if prior is None:
                generator.add_training_data(seg)
                
        ## generate prior
        if prior is None:
            from scipy import ndimage
            mask    = generator.get_mask()
            struct  = np.ones((7,)*mask.ndim)
            mask    = ndimage.binary_dilation(
                    mask.astype(bool),
                    structure=struct,
                    )
            ioanalyze.save(file_mask, mask.astype(np.int32))
            prior = generator.get_prior(mask=mask)
            np.savez(file_prior,**prior)
        
        ## make seeds from mask
        seeds = (-1)*mask.astype(int)
        
        ## loss function
        class Loss(object):
            def __init__(self, nlabel):
                self.nlabel = nlabel
            def __call__(self,z,y_):
                ''' 1 - (z.y_)/nnode '''
                nnode = z.size/float(self.nlabel)
                return 1.0 - 1.0/nnode * np.dot(z,y_)
        
        ## psi
        class Psi(object):
            def __init__(self, 
                    prior, labelset, weight_functions, rwparams, **kwargs):
                self.prior = prior
                self.labelset = labelset
                self.weight_functions = weight_functions
                self.seeds = kwargs.pop('seeds', [])
                self.rwparams = rwparams
                
            def __call__(self, x,y):
                ''' - sum(a){Ea(x,y)} '''
                
                ## energy value for each weighting function
                v = []
                for wf in self.weight_functions.values():
                    
                    v.append(
                        rwmean_svm.energy_RW(
                            x,self.labelset,y,
                            weight_function=wf, 
                            seeds=self.seeds,
                            **self.rwparams
                        ))
                        
                ## last coef is prior
                v.append(
                    rwmean_svm.energy_mean_prior(
                        self.prior,y,
                        seeds=self.seeds,
                        **self.rwparams
                    ))
                    
                ## psi[a] = minus energy[a]
                return np.asmatrix(-np.c_[v])
        
        
        ## most violated constraint
        class Most_violated_constraint(object):
            def __init__(
                    self, prior, 
                    weight_functions, 
                    labelset, 
                    rwparams,
                    **kwargs):
                self.prior = prior
                self.weight_functions = weight_functions
                self.labelset = labelset
                self.rwparams = rwparams
                self.seeds = kwargs.pop('seeds', [])
                
            def __call__(self,w,x,z):
                ''' y_ = arg min <w|-psi(x,y_)> - loss(y,y_) '''
                
                ## combine all weight functions
                def wwf(im,_w):    
                    ''' meta weight function'''
                    data = 0
                    for iwf,wf in enumerate(self.weight_functions.values()):
                        ij,_data = wf(im)
                        data += _w[iwf]*_data
                    return ij, data
                
                ## loss as a linear term in the function
                nnode = x.size
                labelset = np.asarray(self.labelset)
                linloss = 1./nnode*z
                
                ## best y_ most different from y
                y_ = rwmean_svm.segment_mean_prior(
                    x, 
                    self.prior, 
                    seeds=self.seeds,
                    weight_function=lambda im: wwf(im, w),
                    add_linear_term=linloss,
                    lmbda=w[-1],
                    **self.rwparams
                    )
                    
                return y_
        
        ## instanciate functors
        loss = Loss(len(labelset))
        psi = Psi(prior, labelset,weight_functions,rwparams,seeds=seeds)
        mvc = Most_violated_constraint( 
            prior, 
            weight_functions, 
            labelset, 
            rwparams,
            seeds=seeds,
            )
        
        ## learn struct svm
        print 'start learning'
        svm = structsvmpy.StructSVM(
            training_set,
            loss,
            psi,
            mvc,
            **svmparams
            )
            
        ##
        w,xi,info = svm.train()
        np.savetxt(outdir + 'w',w)
        np.savetxt(outdir + 'xi',[xi])
        
        import ipdb; ipdb.set_trace()
        ## segment test image with trained w
        
        
        ## end process
        
    for test in ['01/']:
    # for test in config.vols:
        process(test)
    
    

    
    
if __name__=='__main__':
    ''' start script '''
    main()