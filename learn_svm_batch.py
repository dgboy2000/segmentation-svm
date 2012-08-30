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

# import commands
import sys
import os
import subprocess

import numpy as np

## Initialize logging before loading any modules
import config
reload(config)

from rwsegment import io_analyze
from rwsegment import weight_functions as wflib
from rwsegment import rwsegment
from rwsegment import rwsegment_prior_models as models
from rwsegment import struct_svm
from rwsegment import latent_svm
from rwsegment.rwsegment import BaseAnchorAPI
reload(rwsegment),
reload(wflib)
reload(models)
reload(struct_svm)
reload(latent_svm)

from rwsegment import svm_worker
reload(svm_worker)

from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef


import svm_rw_api
reload(svm_rw_api)
from svm_rw_api import SVMRWMeanAPI
from svm_rw_api import MetaAnchor




from rwsegment import utils_logging
logger = utils_logging.get_logger('learn_svm_batch',utils_logging.DEBUG)

class SVMSegmenter(object):

    def __init__(self,
            use_parallel=True, 
            use_latent=False,
            loss_type='anchor',
            ntrain='all',
            ):
    
        ## paths
        self.dir_reg = config.dir_reg
        self.dir_inf = config.dir_inf
        self.dir_svm = config.dir_svm
        
        ## params
        self.use_latent = use_latent
        self.retrain = True
        self.force_recompute_prior = False
        self.use_parallel = use_parallel

        
        ## params
        # slices = [slice(20,40),slice(None),slice(None)]
        slices = [slice(None),slice(None),slice(None)]
        
        self.labelset = np.asarray([0,13,14,15,16])
        
        if ntrain in ['all']:
            self.training_vols = config.vols
        elif ntrain.isdigit():
            n = int(ntrain)
            self.training_vols = config.vols.keys()[:n]
        # self.training_vols = ['02/'] ## debug
        # self.training_vols = ['02/','03/'] ## debug
        # self.training_vols = config.vols

        
        ## parameters for rw learning
        self.rwparams_svm = {
            'labelset':self.labelset,
            
            # optimization
            'rtol': 1e-6,#1e-5,
            'maxiter': 1e3,
            'per_label':True,
            'optim_solver':'unconstrained',
            }
        
        ## parameters for svm api
        self.svm_api_params = {
            'loss_type': loss_type,#'laplacian',#'anchor',
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
            
            
            # latent
            'latent_niter_max': 100,
            'latent_C': 10,
            'latent_epsilon': 1e-3,
            'latent_use_parallel': self.use_parallel,
            }
            
        ## weight functions
        self.weight_functions = {
            'std_b10'     : lambda im: wflib.weight_std(im, beta=10),
            'std_b50'     : lambda im: wflib.weight_std(im, beta=50),
            'std_b100'    : lambda im: wflib.weight_std(im, beta=100),
            'inv_b100o1'  : lambda im: wflib.weight_inv(im, beta=100, offset=1),
            # 'pdiff_r1b10': lambda im: wflib.weight_patch_diff(im, r0=1, beta=10),
            # 'pdiff_r2b10': lambda im: wflib.weight_patch_diff(im, r0=2, beta=10),
            # 'pdiff_r1b50' : lambda im: wflib.weight_patch_diff(im, r0=1, beta=50),
            }
        
        
        
        ## priors models
        self.prior_models = {
            'constant': models.Constant,
            'entropy': models.Entropy_no_D,
            'intensity': models.Intensity,
            }

        ## indices of w
        nlaplacian = len(self.weight_functions)
        nprior = len(self.prior_models)
        self.indices_laplacians = np.arange(nlaplacian)
        self.indices_priors = np.arange(nlaplacian,nlaplacian + nprior)
        
        ## compute the scale of psi
        self.psi_scale = [1e4] * nlaplacian + [1e5] * nprior
        self.svmparams['psi_scale'] = self.psi_scale
        
        ## make arrays of function
        self.laplacian_functions = self.weight_functions.values()
        self.laplacian_names     = self.weight_functions.keys()
        self.prior_functions     = self.prior_models.values()
        self.prior_names         = self.prior_models.keys()
        
        ## parallel ?
        if self.use_parallel:
            ## communicator
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.MPI_rank = self.comm.Get_rank()
            self.MPI_size = self.comm.Get_size()
            self.isroot = self.MPI_rank==0
            if self.MPI_size==1:
                logger.warning('Found only one process. Not using parallel')
                self.use_parallel = False
            self.svmparams['use_parallel'] = self.use_parallel
            self.svmparams['latent_use_parallel'] = self.use_parallel
        else:
            self.isroot = True
            
        if self.isroot:
            logger.info('passed these command line arguments: {}'.format(str(sys.argv)))
            logger.info('using parallel?: {}'.format(use_parallel))
            logger.info('using latent?: {}'.format(use_latent))
            strkeys = ', '.join(self.laplacian_names)
            logger.info('laplacian functions (in order):{}'.format(strkeys))
            strkeys = ', '.join(self.prior_names)
            logger.info('prior models (in order):{}'.format(strkeys))
            logger.info('using loss type:{}'.format(loss_type))
            logger.info('writing svm output to: {}'.format(self.dir_svm))
        
        
    def train_svm(self,test):
        outdir = test

        ## training images and segmentations
        self.training_set = []
        if self.isroot:
            logger.info('Learning with {} training examples'\
            .format(len(self.training_vols)))
            for train in self.training_vols:
                if test==train: continue
                if self.isroot:  
                    logger.info('loading training data: {}'.format(train))
                file_seg = self.dir_reg + test + train + 'regseg.hdr'
                file_im  = self.dir_reg + test + train + 'reggray.hdr'
                
                im  = io_analyze.load(file_im)
                im = im/np.std(im) # normalize image by std
                
                seg = io_analyze.load(file_seg).astype(int)
                seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]
                z = (seg.ravel()==np.c_[self.labelset])# make bin vector z
                
                self.training_set.append((im, z))

        ## instantiate functors
        
        self.svm_rwmean_api = SVMRWMeanAPI(
            self.prior, 
            self.laplacian_functions, 
            self.labelset,
            self.rwparams_svm,
            prior_models=self.prior_functions,   
            seeds=self.seeds,
            **self.svm_api_params
            )
        
        
        if self.isroot:
            try:
                ## learn struct svm
                logger.debug('started root learning')
                if self.use_latent:
                    self.svm = latent_svm.LatentSVM(
                        self.svm_rwmean_api.compute_loss,
                        self.svm_rwmean_api.compute_psi,
                        self.svm_rwmean_api.compute_mvc,
                        self.svm_rwmean_api.compute_aci,
                        **self.svmparams
                        )
                    w,xi,info = self.svm.train(self.training_set)
                
                else:
                    self.svm = struct_svm.StructSVM(
                        self.training_set,
                        self.svm_rwmean_api.compute_loss,
                        self.svm_rwmean_api.compute_psi,
                        self.svm_rwmean_api.compute_mvc,
                        wsize=self.svm_rwmean_api.wsize,
                        **self.svmparams
                        )
                    w,xi,info = self.svm.train()
                
            except Exception as e:
                import traceback
                logger.error('{}: {}'.format(e.message, e.__class__.__name__))
                traceback.print_exc()
                import ipdb; ipdb.set_trace()
            finally:
                ##kill signal
                if self.use_parallel:
                    # self.comm.bcast(('stop',None),root=0)
                    logger.info('root finished training svm on {}. about to kill workers'\
                        .format(test))
                    for n in range(1, self.MPI_size):
                        logger.debug('sending kill signal to worker #{}'.format(n))
                        self.comm.send(('stop',None),dest=n)
                return w,xi,info
                
        else:
            ## parallel loss augmented inference
            rank = self.MPI_rank
            logger.debug('started worker #{}'.format(rank))
            
            worker = svm_worker.SVMWorker(self.svm_rwmean_api)
            worker.work()
            logger.debug('worker #{} about to exit'.format(rank))
            sys.exit(0)
        
        
        
        
    def run_svm_inference(self,test,w):
        logger.info('running inference on: {}'.format(test))
        
        ## normalize w
        # w = w / np.sqrt(np.dot(w,w))
        strw = ' '.join('{:.3}'.format(val) for val in np.asarray(w)*self.psi_scale)
        logger.debug('scaled w=[{}]'.format(strw))
        
        outdir = self.dir_inf + test
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
    
        weights_laplacians = np.asarray(w)[self.indices_laplacians]
        weights_priors = np.asarray(w)[self.indices_priors]
    
        ## segment test image with trained w
        def meta_weight_functions(im,_w):    
            ''' meta weight function'''
            data = 0
            for iwf,wf in enumerate(self.laplacian_functions):
                ij,_data = wf(im)
                data += _w[iwf]*_data
            return ij, data
        weight_function = lambda im: meta_weight_functions(im, weights_laplacians)
        
        ## load images and ground truth
        file_seg = self.dir_reg + test + 'seg.hdr'
        file_im  = self.dir_reg + test + 'gray.hdr'
        im  = io_analyze.load(file_im)
        seg = io_analyze.load(file_seg)
        seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]
        
        ## save image
        io_analyze.save(outdir + 'im.hdr',im.astype(np.int32))
        im = im/np.std(im) # normalize image by std
    
        ## prior
        # anchor_api = BaseAnchorAPI(
            # self.prior, 
            # anchor_weight=w[-1],
            # )
        anchor_api = MetaAnchor(
            self.prior,
            self.prior_functions,
            weights_priors,
            image=im,
            )
    
        sol,y = rwsegment.segment(
            im, 
            anchor_api, 
            seeds=self.seeds,
            weight_function=weight_function,
            **self.rwparams_inf
            )
        
        np.save(outdir + 'y.npy',y)        
        io_analyze.save(outdir + 'sol.hdr',sol.astype(np.int32))
        
        ## compute Dice coefficient
        dice = compute_dice_coef(sol, seg,labelset=self.labelset)
        np.savetxt(
            outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %f')
        
        
    def process_sample(self, test):
        outdir = self.dir_svm + test
        
        if self.isroot:
            logger.info("output directory: {}".format(outdir))
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            
            prior, mask = load_or_compute_prior_and_mask(
                test,force_recompute=self.force_recompute_prior)
            
            if self.use_parallel:
                # have only the root process compute the prior 
                # and pass it to the other processes
                self.comm.bcast((dict(prior.items()),mask),root=0)    
        else:
            prior,mask = self.comm.bcast(None,root=0)
        
        self.prior = prior
        self.seeds = (-1)*mask.astype(int)
        
        ## training
        if self.retrain:
            w,xi,info = self.train_svm(test)
            np.savetxt(outdir + 'w',w)
            np.savetxt(outdir + 'xi',[xi])     
        else:
            if self.isroot and not self.retrain:    
                logger.warning('Not retraining svm')
            w = np.loadtxt(outdir + 'w')
        
        self.w = w
        
        ## inference
        if self.isroot: 
            self.run_svm_inference(test,w)
        
        
    
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
    
##------------------------------------------------------------------------------



    
    
if __name__=='__main__':
    from optparse import OptionParser
    opt = OptionParser()
    opt.add_option( # use parallet
        '-p', '--parallel', dest='parallel', 
        action="store_true", default=False,
        help='use parallel',
        )
    opt.add_option( # use latent
        '-l', '--latent', dest='latent', 
        action="store_true", default=False,
        help='latent svm',
        )
    opt.add_option( # loss type
        '-o', '--loss', dest='loss', 
        default='anchor', type=str,
        help='loss type ("anchor", "laplacian")',
        )
    opt.add_option( # nb training set
        '-t', '--training', dest='ntrain', 
        default='all', type=str,
        help='number of training set (default: "all")',
        )
        
    (options, args) = opt.parse_args()
    use_parallel = bool(options.parallel)
    use_latent = bool(options.latent)
    loss_type = options.loss
    ntrain = options.ntrain

    ''' start script '''
    svm_segmenter = SVMSegmenter(
        use_parallel=use_parallel,
        use_latent=use_latent,
        loss_type=loss_type,
        ntrain=ntrain,
        )
    sample_list = ['01/']
    
    svm_segmenter.process_all_samples(sample_list)

    
