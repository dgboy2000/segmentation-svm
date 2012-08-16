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

import numpy as np

from rwsegment import io_analyze
from rwsegment import weight_functions as wflib
from rwsegment import rwsegment
from rwsegment import rwsegment_prior_models
from rwsegment import struct_svm
reload(rwsegment),
reload(wflib)
reload(rwsegment_prior_models)
reload(struct_svm)
from rwsegment.rwsegment import BaseAnchorAPI

from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef


import svm_rw_api
reload(svm_rw_api)
from svm_rw_api import SVMRWMeanAPI

## load volume names 
from test import config
reload(config)



class SVMSegmenter(object):

    def __init__(self):
    
        ## paths
        self.dir_reg = config.dir_reg
        self.dir_inf = config.dir_work + 'learning/inference/'
        self.dir_svm = config.dir_work + 'learning/svm/'
        
        ## re-train svm?
        self.retrain = True
        self.force_recompute_prior = False
        
        ## params
        # slices = [slice(20,40),slice(None),slice(None)]
        slices = [slice(None),slice(None),slice(None)]
        
        self.labelset = np.asarray([0,13,14,15,16])
        
        self.training_vols = ['02/'] ## debug
        # self.training_vols = config.vols
        
        ## parameters for rw learning
        self.rwparams_svm = {
            'labelset':self.labelset,
            
            # optimization
            'rtol': 1e-6,
            'maxiter': 1e3,
            'per_label':False,
            # 'optim_solver':'unconstrained', #debug
            'optim_solver':'constrained',
            }
            
        ## parameters for rw inference
        self.rwparams_inf = {
            'labelset':self.labelset,
            'return_arguments':['image','y'],
            
            # optimization
            'rtol': 1e-6,
            'maxiter': 1e3,
            'per_label':True,
            'optim_solver':'unconstrained',
            }
            
        ## svm params
        self.svmparams = {
            'C': 1,
            'nitermax':100,
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
        
        
    def train_svm(self,test):
        outdir = test

        ## training images and segmentations
        self.training_set = []
        for train in self.training_vols:
            if test==train: continue
            logger.info('loading training data: {}'.format(train))
            file_seg = self.dir_reg + test + train + 'regseg.hdr'
            file_im  = self.dir_reg + test + train + 'reggray.hdr'
            
            im  = io_analyze.load(file_im)
            im = im/np.std(im) # normalize image by std
            
            seg = io_analyze.load(file_seg)
            seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]
            z = (seg.ravel()==np.c_[self.labelset])# make bin vector z
            
            self.training_set.append((im, z))

        ## instanciate functors
        self.svm_rwmean_api = SVMRWMeanAPI(
            self.prior, 
            self.weight_functions, 
            self.labelset, 
            self.rwparams_svm,
            seeds=self.seeds,
            )
        
        ## learn struct svm
        logger.debug('start learning')
        self.svm = struct_svm.StructSVM(
            self.training_set,
            self.svm_rwmean_api.compute_loss,
            self.svm_rwmean_api.compute_psi,
            self.svm_rwmean_api.compute_mvc,
            **self.svmparams
            )

        w,xi,info = self.svm.train()
        
        return w,xi,info
        
        
    def run_svm_inference(self,test,w):
        logger.info('running inference on: {}'.format(test))
        
        outdir = self.dir_inf + test
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
    
        ## segment test image with trained w
        def wwf(im,_w):    
            ''' meta weight function'''
            data = 0
            for iwf,wf in enumerate(self.weight_functions.values()):
                ij,_data = wf(im)
                data += _w[iwf]*_data
            return ij, data
        
        ## load images and ground truth
        file_seg = self.dir_reg + test + 'seg.hdr'
        file_im  = self.dir_reg + test + 'gray.hdr'
        im  = io_analyze.load(file_im)
        seg = io_analyze.load(file_seg)
        seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]
        
        ## save image
        im = im/np.std(im) # normalize image by variance
    
        ## prior
        anchor_api = BaseAnchorAPI(
            self.prior, 
            anchor_weight=w[-1],
            )
    
        sol,y = rwsegment.segment(
            im, 
            anchor_api, 
            seeds=self.seeds,
            weight_function=lambda im: wwf(im, w),
            **self.rwparams_inf
            )
        
        np.save(outdir + 'y.test.npy',y)        
        io_analyze.save(outdir + 'sol.test.hdr',sol.astype(np.int32))
        
        ## compute Dice coefficient
        dice = compute_dice_coef(sol, seg,labelset=self.labelset)
        np.savetxt(
            outdir + 'dice.test.txt', np.c_[dice.keys(),dice.values()],fmt='%d %.8f')
            
        ## inference compare with gold standard
        dice_gold = np.loadtxt(outdir + 'dice.gold.txt')
        y_gold    = np.load(outdir + 'y.gold.npy')        
        sol_gold  = io_analyze.load(outdir + 'sol.gold.hdr')
        
        np.testing.assert_allclose(
            dice.values(), 
            dict(dice_gold).values(), 
            err_msg='FAIL: dice coef mismatch',
            atol=1e-8)
        np.testing.assert_allclose(y, y_gold,  err_msg='FAIL: y mismatch')
        np.testing.assert_equal(sol, sol_gold, err_msg='FAIL: sol mismatch')
        
        print 'PASS: inference tests'
        
    def process_sample(self, test):
        outdir = self.dir_svm + test
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        prior, mask = load_or_compute_prior_and_mask(test, force_recompute=self.force_recompute_prior)
        
        self.prior = prior
        self.seeds = (-1)*mask.astype(int)
        
        ## training
        if self.retrain:
            w,xi,info = self.train_svm(test)
            np.savetxt(outdir + 'w.test.txt',w)
            np.savetxt(outdir + 'xi.test.txt',[xi])
            
            try:
                w_gold  = np.loadtxt(outdir + 'w.gold.txt')        
                xi_gold = np.loadtxt(outdir + 'xi.gold.txt')        
                
                np.testing.assert_allclose(w,  w_gold,  err_msg='w mismatch')
                np.testing.assert_allclose(xi, xi_gold, err_msg='xi mismatch')
            except Exception as e:
                print str(type(e)) + ', ' + e.message
            else:
                print 'PASS: learning tests'
            
        else:
            w = np.loadtxt(outdir + 'w.test.txt')
        
        self.w = w
        
        ## inference
        self.run_svm_inference(test,w)

        
    
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
    
##------------------------------------------------------------------------------

from rwsegment import utils_logging
logger = utils_logging.get_logger('learn_svm_batch',utils_logging.DEBUG)

    
    
if __name__=='__main__':
    ''' start script '''
    svm_segmenter = SVMSegmenter()
    sample_list = ['01/']
    # sample_list = config.vols
    svm_segmenter.process_all_samples(sample_list)