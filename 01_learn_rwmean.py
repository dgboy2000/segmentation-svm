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

import svm_rw_api
reload(svm_rw_api)
from svm_rw_api import Loss, Psi, Most_violated_constraint

import structsvmpy 
reload(structsvmpy)

def main():
    ## load volume names 
    import config
    reload(config)
    
    ## paths
    dir_reg     = config.dir_reg
    
    ## re-train svm?
    retrain = True
    
    ## params
    # slices = [slice(20,40),slice(None),slice(None)]
    slices = [slice(None),slice(None),slice(None)]
    
    labelset = np.asarray([0,13,14,15,16])
    
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
        # 'pdiff_r1b50' : lambda im: wflib.weight_patch_diff(im, r0=1, beta=50),
        # 'pdiff_r1b100': lambda im: wflib.weight_patch_diff(im, r0=1, beta=100),
        }
        
    def load_or_compute_prior_and_mask(test):
        outdir = test
        
        ## load mask and prior
        prior = None
        file_mask  = outdir + 'mask.hdr'
        file_prior = outdir + 'prior.npz'
        if os.path.exists(file_prior):
            logger.info('load prior')
            mask  = ioanalyze.load(file_mask)
            prior = np.load(file_prior)
        else:
            logger.info('compute prior')
            generator = rwmean_svm.PriorGenerator(labelset)
            for train in config.vols:
                if test==train: continue
                file_seg = dir_reg + test + train + 'regseg.hdr'
                seg = ioanalyze.load(file_seg)
                generator.add_training_data(seg)
            
            from scipy import ndimage
            mask    = generator.get_mask()
            struct  = np.ones((7,)*mask.ndim)
            mask    = ndimage.binary_dilation(
                    mask.astype(bool),
                    structure=struct,
                    )
            prior = generator.get_prior(mask=mask)
            np.savez(file_prior,**prior)
            ioanalyze.save(file_mask, mask.astype(np.int32))
            
        
                
                
        return prior, mask
        
    def train_svm(test, prior, mask):
        outdir = test

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
                
        ## make seeds from mask
        seeds = (-1)*mask.astype(int)

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
       
        # import ipdb; ipdb.set_trace()
        
        return w,xi,info
        
        
    def run_svm_inference(test,w,prior, mask):
    
        outdir = test
    
        ## segment test image with trained w
        def wwf(im,_w):    
            ''' meta weight function'''
            data = 0
            for iwf,wf in enumerate(weight_functions.values()):
                ij,_data = wf(im)
                data += _w[iwf]*_data
            return ij, data
        
        file_seg = dir_reg + test + 'seg.hdr'
        file_im  = dir_reg + test + 'gray.hdr'
        im  = ioanalyze.load(file_im)
        seg = ioanalyze.load(file_seg)
        
        ## make bin vector z from segmentation
        seg.flat[~np.in1d(seg.ravel(),labelset)] = labelset[0]
        # bin = (np.c_[seg.ravel()]==labelset).ravel('F')
        
        ## save image
        ioanalyze.save(outdir + 'im.hdr',im.astype(int))
        
        ## normalize image by variance
        im = im/np.std(im)
    
        ## make seeds from mask
        seeds = (-1)*mask.astype(int)
    
        y = rwmean_svm.segment_mean_prior(
            im, 
            prior, 
            seeds=seeds,
            weight_function=lambda im: wwf(im, w),
            lmbda=w[-1],
            **rwparams
            )
        
        np.save(outdir + 'y.npy',y)
        
        sol = labelset[
            np.argmax(y.reshape((-1,len(labelset)),order='F'),axis=1)]\
            .reshape(im.shape)
        
        
        ioanalyze.save(outdir + 'sol.hdr',sol)
        
        
        ## Dice coef(sol, seg)
        
        
    def process(test, retrain=True):
        ## learn rwmean parameters
        
        outdir = test
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        prior, mask = load_or_compute_prior_and_mask(test)
        
        if retrain:
            w,xi,info = train_svm(test,prior,mask)
            np.savetxt(outdir + 'w',w)
            np.savetxt(outdir + 'xi',[xi])            
        else:
            w = np.loadtxt(outdir + 'w')
            
        run_svm_inference(test,w,prior,mask)

        ## end process
        
    for test in ['01/']:
    # for test in config.vols:
        process(test,retrain)
    
##------------------------------------------------------------------------------
    
import logging
logger = logging.getLogger('svm-segmentation logger')
loglevel = logging.INFO
logger.setLevel(loglevel)
# create console handler with a higher log level
if len(logger.handlers)==0:
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
else:
    logger.handlers[0].setLevel(loglevel)

    
    
if __name__=='__main__':
    ''' start script '''
    main()