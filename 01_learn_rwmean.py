'''
    Notes:
        - the energy should only be computed wrt to the unkwnown pixels.
        In some place we sloppily include known pixels in the computation.
        These do not matter since a constant does not change the solution.

'''


import sys
import os
import logging

import numpy as np

##temp: load rwsegment lib
sys.path += [os.path.abspath('../')]

from rwsegment import ioanalyze
from rwsegment import weight_functions as wf
from rwsegment import rwmean_svm
reload(rwmean), reload(wf)

import svmStruct 
reload(svmStruct)

def main():
    ## load volume names 
    import config
    
    ## paths
    dir_reg     = config.dir_reg
    dir_prior   = config.dir_prior
    
    ## params
    labelset = [0,13,14,15,16]
    params = {
        ## rw params
        'labelset':labelset,
        'tol': 1e-3,
        'maxiter': 1e2,
        
        ## svm params
        'C': 100,
        'nitermax':100,
        'loglevel':logging.INFO,
        }
        
    ## weight functions
    weight_functions = {
        'std_b10'     : lambda im: wfunc.weight_std(im, beta=10),
        # 'std_b50'     : lambda im: wfunc.weight_std(im, beta=50),
        # 'std_b100'    : lambda im: wfunc.weight_std(im, beta=100),
        # 'inv_b100o1'  : lambda im: wfunc.weight_inv(im, beta=100, offset=1),
        'pdiff_r1b50' : lambda im: wfunc.weight_patch_diff(im, r0=1, beta=50),
        # 'pdiff_r1b100': lambda im: wfunc.weight_patch_diff(im, r0=1, beta=100),
        }
        
        
    def process(test):
        ## learn rwmean parameters
        
        outdir = test
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        ## load mask and prior
        print 'load mask and prior'
        file_mask  = dir_prior + test + 'mask.hdr'
        file_prior = dir_prior + test + 'prior.npz'
        mask  = ioanalyze.load(file_mask)
        prior = np.load(file_prior)
        
        seeds = (-1)*mask.astype(int))
        
        ## training images and segmentations
        training_set = []
        for train in config.vols:
            if test==train: continue
            print '  load training data: {}'.format(train)
            
            file_seg = dir_reg + test + train + 'regseg.hdr'
            file_im  = dir_reg + test + train + 'reggray.hdr'
            im  = ioanalyze.load(file_im)
            seg = ioanalyze.load(file_seg)
            
            bin = (np.c_[seg.ravel()]==labelset).ravel('F')
            
            training_set.append((im, bin))
        

        
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
                    prior, weight_functions, params, **kwargs):
                self.prior = prior
                self.weight_functions = weight_functions
                self.seeds = kwargs.pop('seeds', [])
                self.params = params
                
            def __call__(self, x,y):
                ''' - sum(a){Ea(x,y)} '''
                
                ## energy value for each weighting function
                v = []
                for wf in self.weight_functions:
                    v.append(
                        rwmean_svm.energy_RW(
                            x,y,
                            weight_function=wf, 
                            seeds=self.seeds,
                            **self.params
                        ))
                        
                ## last coef is prior
                v.append(
                    rwmean_svm.energy_mean_prior(
                        self.prior,y,
                        seeds=self.seeds,
                        **self.params
                    ))
                    
                ## psi[a] = minus energy[a]
                return np.asmatrix(-np.c_[v])
        
        
        ## most violated constraint
        class Most_violated_constraint(object):
            def __init__(
                    self, prior, 
                    weight_functions, 
                    labelset, 
                    params,
                    **kwargs):
                self.prior = prior
                self.weight_functions = weight_functions
                self.labelset = labelset
                self.params = params
                self.seeds = kwargs.pop('seeds', [])
                
            def __call__(self,w,x,z):
                ''' y_ = arg min <w|-psi(x,y_)> - loss(y,y_) '''
                
                ## combine all weight functions
                def wwf(im):    
                    ''' meta weight function'''
                    data = 0
                    for iwf, wf in enumerate(self.weight_functions):
                        ij,_data = wf(im)
                        data += w[iwf]*_data
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
                    weight_function=wwf,
                    add_linear_term=linloss,
                    **self.params
                    )
                    
                return y_
        
        
        loss = Loss(len(labelset))
        psi = Psi(prior,weight_functions,params,seeds=seeds)
        mvc = Most_violated_constraint( 
            prior, 
            weight_functions, 
            labelset, 
            params,
            seeds=seeds,
            )
        
        ## learn struct svm
        svm = StructSVM(
            training_set,
            loss,
            psi,
            mvc,
            **params
            )
            
        w,xi,info = svm.train()
        np.savetxt(outdir + 'w',w)
        np.savetxt(outdir + 'xi',xi)
        
        ## segment test image with trained w
        
        
        ## end process
        
    for test in ['01/']:
    # for test in config.vols:
        process(test)
    
    

    
    
if __name__=='__main__':
    ''' start script '''
    main()