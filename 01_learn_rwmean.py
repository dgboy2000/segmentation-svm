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
from segmentation_utils import load_or_compute_prior_and_mask
reload(rwmean_svm), reload(wflib)

import svm_rw_api
reload(svm_rw_api)
from svm_rw_api import SVMRWMeanAPI

import structsvmpy 
reload(structsvmpy)

## load volume names 
import config
reload(config)

rwmean_svm.logger.setLevel(logging.DEBUG)
rwmean_svm.logger.handlers[0].setLevel(logging.DEBUG)

class SVMSegmenter(object):

    def __init__(self):
    
        ## paths
        self.dir_reg     = config.dir_reg
        
        ## re-train svm?
        # retrain = False
        self.retrain = True
        
        ## params
        # slices = [slice(20,40),slice(None),slice(None)]
        slices = [slice(None),slice(None),slice(None)]
        
        self.labelset = np.asarray([0,13,14,15,16])
        
        ## rw params        
        self.rwparams = {
            'labelset':self.labelset,
            'rtol': 1e-6,
            'maxiter': 1e3,
            'per_label':True,
            'optim_solver':'scipy',
            'return_arguments':['y'],
            }
            
        ## svm params
        self.svmparams = {
            'C': 1,
            'nitermax':100,
            # 'loglevel':logging.INFO,
            'loglevel':logging.DEBUG,
            }
            
        ## weight functions
        self.weight_functions = {
            'std_b10'     : lambda im: wflib.weight_std(im, beta=10),
            # 'std_b50'     : lambda im: wflib.weight_std(im, beta=50),
            # 'std_b100'    : lambda im: wflib.weight_std(im, beta=100),
            # 'inv_b100o1'  : lambda im: wflib.weight_inv(im, beta=100, offset=1),
            # 'pdiff_r1b50' : lambda im: wflib.weight_patch_diff(im, r0=1, beta=50),
            # 'pdiff_r1b100': lambda im: wflib.weight_patch_diff(im, r0=1, beta=100),
            }
        
        
    def train_svm(self,test, prior, mask):
        outdir = test

        ## training images and segmentations
        self.training_set = []
        for train in config.vols:
            if test==train: continue
            print '  load training data: {}'.format(train)
            file_seg = self.dir_reg + test + train + 'regseg.hdr'
            file_im  = self.dir_reg + test + train + 'reggray.hdr'
            im  = ioanalyze.load(file_im)
            seg = ioanalyze.load(file_seg)
            
            ## make bin vector z from segmentation
            seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]
            bin = (np.c_[seg.ravel()]==self.labelset).ravel('F')
            
            ## normalize image by variance
            im = im/np.std(im)
            
            self.training_set.append((im, bin))
                
        ## make seeds from mask
        seeds = (-1)*mask.astype(int)

        ## instanciate functors
        self.svm_rwmean_api = SVMRWMeanAPI(
            prior, 
            self.weight_functions, 
            self.labelset, 
            self.rwparams,
            seeds=seeds,
            )
        
        ## learn struct svm
        print 'start learning'
        self.svm = structsvmpy.StructSVM(
            self.training_set,
            self.svm_rwmean_api.compute_loss,
            self.svm_rwmean_api.compute_psi,
            self.svm_rwmean_api.compute_mvc,
            **self.svmparams
            )
            
        w,xi,info = self.svm.train()
       
        # import ipdb; ipdb.set_trace()
        
        return w,xi,info
        
        
    def run_svm_inference(self,test,w,prior, mask):
    
        outdir = test
    
        ## segment test image with trained w
        def wwf(im,_w):    
            ''' meta weight function'''
            data = 0
            for iwf,wf in enumerate(self.weight_functions.values()):
                ij,_data = wf(im)
                data += _w[iwf]*_data
            return ij, data
        
        file_seg = self.dir_reg + test + 'seg.hdr'
        file_im  = self.dir_reg + test + 'gray.hdr'
        im  = ioanalyze.load(file_im)
        seg = ioanalyze.load(file_seg)
        
        ## make bin vector z from segmentation
        seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]
        
        ## save image
        ioanalyze.save(outdir + 'im.hdr',im.astype(int))
        
        ## normalize image by variance
        im = im/np.std(im)
    
        ## make seeds from mask
        seeds = (-1)*mask.astype(int)
    
        # import ipdb; ipdb.set_trace()
        y = rwmean_svm.segment_mean_prior(
            im, 
            prior, 
            seeds=seeds,
            weight_function=lambda im: wwf(im, w),
            lmbda=w[-1],
            **self.rwparams
            )
        
        np.save(outdir + 'y.npy',y)
        
        nlabel = len(self.labelset)
        sol = self.labelset[
            np.argmax(y.reshape((-1,nlabel),order='F'),axis=1)]\
            .reshape(im.shape)
        
        ioanalyze.save(outdir + 'sol.hdr',sol)
        
        
        ## Dice coef(sol, seg)
        
        
    def process_sample(self, test):
        ## learn rwmean parameters
        
        outdir = test
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        prior, mask = load_or_compute_prior_and_mask(test)
        
        if self.retrain:
            w,xi,info = self.train_svm(test,prior,mask)
            np.savetxt(outdir + 'w',w)
            np.savetxt(outdir + 'xi',[xi])            
        else:
            w = np.loadtxt(outdir + 'w')
        
        self.w = w
        self.xi = xi
        
        self.run_svm_inference(test,w,prior,mask)
        ## end process
    
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
    
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
    svm_segmenter = SVMSegmenter()
    sample_list = ['01/']
    # sample_list = config.vols
    svm_segmenter.process_all_samples(sample_list)