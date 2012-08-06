import sys
import os

import numpy as np

##temp: load rwsegment lib
sys.path += [os.path.abspath('../')]

from rwsegment import ioanalyze
from rwsegment import weight_functions as wf
from rwsegment import rwmean
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
    params = {
        ## rw params
        labelset=[0,13,14,15,16],
        # 'lmbda': 1e-2, ## to learn !
        'tol': 1e-3,
        'maxiter': 1e2,
        
        ## svm params
        'w_loss': 1e2,
        'C': 5000,
        }
        
    ## weight functions
    weight_functions = {
        'std_b10'     : lambda im: wfunc.weight_std(im, beta=10),
        'std_b50'     : lambda im: wfunc.weight_std(im, beta=50),
        'std_b100'    : lambda im: wfunc.weight_std(im, beta=100),
        'inv_b100o1'  : lambda im: wfunc.weight_inv(im, beta=100, offset=1),
        'pdiff_r1b50' : lambda im: wfunc.weight_patch_diff(im, r0=1, beta=50),
        'pdiff_r1b100': lambda im: wfunc.weight_patch_diff(im, r0=1, beta=100),
        }
        
        
    def process(test):
        ## learn rwmean parameters
        
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
            if vol==train: continue
            print '  load training data: {}'.format(train)
            
            file_seg = dir_reg + test + train + 'regseg.hdr'
            file_im  = dir_reg + test + train + 'reggray.hdr'
            im  = ioanalyze.load(file_im)
            seg = ioanalyze.load(file_seg)
            
            training_set.append((im, seg))
        
        
        ## loss function
        def loss_function(y,y_):
            ''' (-1)(2X-1)t(2X_-1) '''
            n = y.size
            return float(2*np.sum((y!=y_).astype(int)) - n)
        
        ## psi
        def psi(x,y):
            ''' - sum(a){Ea(x,y)} '''
            
            ## energy value for each weighting function
            v = []
            for wf in weighting_functions:
                v.append(
                    rwmean.energy_RW(
                        x,y,
                        weight_function=wf, 
                        seeds=seeds,
                        **params
                    ))
                    
            ## last coef is prior
            v.append(
                rwmean.energy_mean_prior(
                    prior,y,
                    seeds=seeds,
                    **params
                ))
                
            ## psi[a] = minus energy[a]
            return np.asmatrix(-np.c_[v])
        
        
        ## most violated constraint
        def most_violated_constraint(w,x,y):
            ''' y_ = arg min <w|-psi(x,y_)> - loss(y,y_) '''
            
            ## combine all weight functions
            def wwf(im):    
                ''' meta weight function'''
                data = 0
                for iwf, wf in enumerate(weight_functions):
                    ij,_data = wf(x)
                    data += w[iwf]*wf
                return ij, wwf
            
            ## loss as a linear term in the function
            ## loss(X,X_) = -(2X_-1)t(2X-1)
            labelset = np.asarray(params['labelset'])
            linloss = 2*(2*np.asmatrix(np.c_[y]==labelset) - 1.0)
            
            ## best y_ most different from y
            y_ = segment_mean_prior(
                x, 
                prior, 
                seeds=seeds,
                weight_function=wwf,
                add_linear_term=linloss,
                **params
                )
                
            return y_
        
        ## learn struct svm
        svm = StructSVM(
            training_set,
            loss_function,
            psi,
            most_violated_contraint,
            **params
            )
        
        
        
        ## end process
        
    for test in ['01/']:
    # for test in config.vols:
        process(test)
    
    

    
    
if __name__=='__main__':
    ''' start script '''
    main()