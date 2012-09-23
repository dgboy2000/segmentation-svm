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
from rwsegment import loss_functions
from rwsegment.rwsegment import BaseAnchorAPI
reload(rwsegment),
reload(wflib)
reload(models)
reload(struct_svm)
reload(latent_svm)
reload(loss_functions)

from rwsegment import svm_worker
reload(svm_worker)

from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef


import svm_rw_api
reload(svm_rw_api)
from svm_rw_api import SVMRWMeanAPI
from svm_rw_api import MetaAnchor

from rwsegment import mpi


from rwsegment import utils_logging
logger = utils_logging.get_logger('learn_svm_batch',utils_logging.DEBUG)

class SVMSegmenter(object):

    def __init__(self,
            use_parallel=True, 
            use_latent=False,
            loss_type='anchor',
            ntrain='all',
            debug=False,
            **kwargs
            ):
    
        ## paths
        self.dir_reg = config.dir_reg
        self.dir_inf = config.dir_inf
        self.dir_svm = config.dir_svm
        
        ## params
        self.use_latent = use_latent
        #self.retrain = True
        self.force_recompute_prior = False
        self.use_parallel = use_parallel    
        self.debug = debug
        self.retrain = kwargs.pop('retrain', True)
        self.nomosek = kwargs.pop('nomosek',False)
        C = kwargs.pop('C',1.0)
        self.minimal_svm = kwargs.pop('minimal', False)
        self.one_iteration = kwargs.pop('one_iteration', False)
        switch_loss = kwargs.pop('switch_loss', False)
        self.start_script = kwargs.pop('start_script', '')       
 
        ## params
        # slices = [slice(20,40),slice(None),slice(None)]
        slices = [slice(None),slice(None),slice(None)]
        
        self.labelset = np.asarray(config.labelset)
        
        if ntrain in ['all']:
            self.training_vols = config.vols
        elif ntrain.isdigit():
            n = int(ntrain)
            self.training_vols = config.vols.keys()[:n]
        
        ## parameters for rw learning
        self.rwparams_svm = {
            'labelset':self.labelset,
            
            # optimization
            'rtol': 1e-6,#1e-5,
            'maxiter': 1e3,
            'per_label':True,
            # 'per_label':False,
            'optim_solver':'unconstrained',
            'logbarrier_mu': 10,
            'logbarrier_initial_t': 1e-2,
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
            'C': C,
            'nitermax': 100,
            'nomosek': self.nomosek,
            'epsilon': 1e-5,
            'do_switch_loss': switch_loss,

            # latent
            'latent_niter_max': 100,
            'latent_C': 10,
            'latent_epsilon': 1e-3,
            'latent_use_parallel': self.use_parallel,
            }
            
        ## weight functions
        if self.minimal_svm:
            self.weight_functions = {'std_b50': lambda im: wflib.weight_std(im, beta=50)}
        else:
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
        if self.minimal_svm:
            self.prior_models = {'constant': models.Constant}
        else:
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
        #self.psi_scale = [1e4] * nlaplacian + [1e5] * nprior
        self.psi_scale = [1.0] * nlaplacian + [1.0] * nprior
        self.svmparams['psi_scale'] = self.psi_scale
 
        ## make arrays of function
        self.laplacian_functions = self.weight_functions.values()
        self.laplacian_names     = self.weight_functions.keys()
        self.prior_functions     = self.prior_models.values()
        self.prior_names         = self.prior_models.keys()
        
        ## parallel ?
        if self.use_parallel:
            ## communicator
            # from mpi4py import MPI
            # self.comm = MPI.COMM_WORLD
            # self.MPI_rank = self.comm.Get_rank()
            # self.MPI_size = self.comm.Get_size()
            self.comm = mpi.COMM
            self.MPI_rank = mpi.RANK
            self.MPI_size = mpi.SIZE
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
            logger.info('laplacian functions (in order): {}'.format(strkeys))
            strkeys = ', '.join(self.prior_names)
            logger.info('prior models (in order): {}'.format(strkeys))
            logger.info('don\'t use mosek ?: {}'.format(self.nomosek))
            logger.info('using loss type: {}'.format(loss_type))
            logger.info('SVM parameters: {}'.format(self.svmparams))
            logger.info('Computing one iteration at a time ?: {}'.format(self.one_iteration))
            if self.debug:
                logger.info('debug mode, no saving')
            else:
                logger.info('writing svm output to: {}'.format(self.dir_svm))
        
        
    def train_svm(self,test,outdir=''):

        ## training images and segmentations
        self.training_set = []
        ntrain = len(self.training_vols)
        if test in self.training_vols:
            ntrain = ntrain - 1
        if self.isroot:
            logger.info('Learning with {} training examples'\
                .format(ntrain))
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
                import time                

                ## learn struct svm
                logger.debug('started root learning')
                if self.use_latent:
                    self.svm = latent_svm.LatentSVM(
                        self.svm_rwmean_api.compute_loss,
                        self.svm_rwmean_api.compute_psi,
                        self.svm_rwmean_api.compute_mvc,
                        self.svm_rwmean_api.compute_aci,
                        one_iteration=self.one_iteration,
                        **self.svmparams
                        )
                    if self.one_iteration:

                        # if we're computing one iteration at a time
                        if os.path.isfile(outdir + 'niter.txt'):
                            niter = np.loadtxt(outdir + 'niter.txt',dtype=int)
                            ys = np.load(outdir + 'ys.npy')
                            w = np.loadtxt(outdir + 'w_{}.txt'.format(niter))
                            
                            curr_iter = niter + 1
                            logger.info('latent svm: iteration {}, with w={}'.format(curr_iter,w))
                            w,xi,info = self.svm.train(self.training_set, niter0=curr_iter, w0=w, ys=ys)
                        else:
                            niter = 1
                            if self.minimal_svm:
                                w0 = [1.0, 1e-2]
                            else:
                                w0 = [1.0, 0.0, 0.0, 0.0, 1e-2, 0.0, 0.0 ]
                            logger.info('latent svm: first iteration. w0 = {}'.format(w0))
                            w,xi,info = self.svm.train(self.training_set, w0=w0)
                           
                        # save output for next iteration
                        if not self.debug and not info['stopped']:
                            niter = info['niter']
                            np.savetxt(outdir + 'niter.txt', [niter], fmt='%d')
                            np.savetxt(outdir + 'w_{}.txt'.format(niter), w)
                            np.save(outdir + 'ys.npy', info['ys'])
                            
                            logger.warning('Exiting program. Run script again to continue.')
                            if self.use_parallel:
                                for n in range(1, self.MPI_size):
                                    #logger.debug('sending kill signal to worker #{}'.format(n))
                                    self.comm.send(('stop',1, {}),dest=n)

                            ## re-run script
                            #curr_time = time.time()
                            #logger.info('elapsed time = {:.2} hours, with {} iterations'\
                            #    .format((curr_time-self.start_time)*3600, niter))
                            ## check if we have the time to run a new iteration
                            #if (curr_time - self.start_time)/(niter) < (3.90 * 3600.0)/(niter + 1) and \
                            #    os.path.isfile(self.start_script):
                            #    os.system('qsub -k oe {}'.format(self.start_script))
                            #else:
                            logger.info('you should run command line: qsub -k oe {}'.format(self.start_script))

                    else:
                        nvar = len(self.indices_laplacians) + len(self.indices_priors)
                        logger.info('latent svm: start all iterations')
                        w,xi,info = self.svm.train(self.training_set, w0=np.ones(nvar))
                
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
                    logger.info('root finished training svm on {}. about to kill workers'\
                        .format(test))
                    for n in range(1, self.MPI_size):
                        logger.debug('sending kill signal to worker #{}'.format(n))
                        self.comm.send(('stop',None,{}),dest=n)
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
        
        
        nim = im/np.std(im) # normalize image by std
    
        ## prior
        anchor_api = MetaAnchor(
            self.prior,
            self.prior_functions,
            weights_priors,
            image=nim,
            )
    
        sol,y = rwsegment.segment(
            nim, 
            anchor_api, 
            seeds=self.seeds,
            weight_function=weight_function,
            **self.rwparams_inf
            )
        
        ## compute Dice coefficient
        dice = compute_dice_coef(sol, seg,labelset=self.labelset)
        logger.info('Dice coefficients: {}'.format(dice))

        ## objective
        en_rw = rwsegment.energy_rw(
            nim, y, seeds=self.seeds,weight_function=weight_function, **self.rwparams_inf)
        en_anchor = rwsegment.energy_anchor(
            nim, y, anchor_api, seeds=self.seeds, **self.rwparams_inf)
        obj = en_rw + en_anchor
        logger.info('Objective = {:.3}'.format(obj))

        
        ## compute losses
        z = seg.ravel()==np.c_[self.labelset]
        mask = self.seeds < 0
        flatmask = mask.ravel()*np.ones((len(self.labelset),1))
        
        ## loss 0 : 1 - Dice(y,z)
        loss0 = loss_functions.ideal_loss(z,y,mask=flatmask)
        logger.info('Tloss = {}'.format(loss0))
        
        ## loss2: squared difference with ztilde
        loss1 = loss_functions.anchor_loss(z,y,mask=flatmask)
        logger.info('SDloss = {}'.format(loss1))
        
        ## loss3: laplacian loss
        loss2 = loss_functions.laplacian_loss(z,y,mask=flatmask)
        logger.info('LAPloss = {}'.format(loss2))

        ## loss4: linear loss
        loss3 = loss_functions.linear_loss(z,y,mask=flatmask)
        logger.info('LINloss = {}'.format(loss3))


        
        ## saving
        if self.debug:
            pass
        elif self.isroot:
            outdir = self.dir_inf + test
            logger.info('saving data in: {}'.format(outdir))
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
                
            io_analyze.save(outdir + 'im.hdr',im.astype(np.int32))
            np.save(outdir + 'y.npy',y)        
            io_analyze.save(outdir + 'sol.hdr',sol.astype(np.int32))
            np.savetxt(outdir + 'objective.txt', [obj])
            np.savetxt(
                outdir + 'dice.txt', 
                np.c_[dice.keys(),dice.values()],fmt='%d %f')
        
            f = open(outdir + 'losses.txt', 'w')
            f.write('ideal_loss\t{}\n'.format(loss0))
            f.write('anchor_loss\t{}\n'.format(loss1))
            f.write('laplacian_loss\t{}\n'.format(loss2))
            f.close()
        
    def process_sample(self, test):
        
        if self.isroot:
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
            
            outdir = self.dir_svm + test
            if not self.debug and not os.path.isdir(outdir):
                os.makedirs(outdir)
                
            w,xi,info = self.train_svm(test,outdir=outdir)
            
            if self.debug:
                pass
            elif self.isroot:
                np.savetxt(outdir + 'w',w)
                np.savetxt(outdir + 'xi',[xi])     
        else:
            if self.isroot and not self.retrain:    
                outdir = self.dir_svm + test
                logger.warning('Not retraining svm')
                w = np.loadtxt(outdir + 'w')
        
        ## inference
        if self.isroot: 
            self.w = w
            
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
        
    opt.add_option( # nb training set
        '-g', '--debug', dest='debug', 
        default=False, action="store_true",
        help='debug mode (no saving)',
        )
    
    opt.add_option( # no mosek
        '--nomosek', dest='nomosek', 
        default=False, action="store_true",
        help='don\'t use mosek',
        )  
    
    opt.add_option( # retrain ?
        '--noretrain', dest='noretrain', 
        default=False, action="store_true",
        help='retrain svm ?',
        )  
 
    opt.add_option( # minimal svm
        '--minimal', dest='minimal', 
        default=False, action="store_true",
        help='minimal svm: one laplacian, one prior ?',
        )  
 
    opt.add_option( # one iteration at a time
        '--one_iter', dest='one_iter', 
        default=False, action="store_true",
        help='compute one iteration at a time (latent only)',
        )
  
    opt.add_option(
        '--switch_loss', dest='switch_loss', 
        default=False, action="store_true",
        help='use approx loss in the end',
        )  
    opt.add_option(
        '--basis', dest='basis',
        default='default', type=str,
        help='',
        ) 

    opt.add_option( # C
        '-C', dest='C', 
        default=1.0, type=float,
        help='C value',
        )  
       
    opt.add_option( # folder name
        '--folder', dest='folder', 
        default='', type=str,
        help='set folder name',
        ) 
    
    opt.add_option(
        '--script', dest='script', 
        default="", type=str,
        help='script file to run this module',
        )  
 

    (options, args) = opt.parse_args()

    use_parallel = bool(options.parallel)
    use_latent = bool(options.latent)
    loss_type = options.loss
    ntrain = options.ntrain
    debug = options.debug
    nomosek = options.nomosek
    retrain = 1 - options.noretrain
    minimal = options.minimal
    one_iteration = options.one_iter
    switch_loss = options.switch_loss
    C = options.C
    script = options.script

    folder = options.folder #unused

    ''' start script '''
    svm_segmenter = SVMSegmenter(
        C=C,
        use_parallel=use_parallel,
        use_latent=use_latent,
        loss_type=loss_type,
        ntrain=ntrain,
        debug=debug,
        retrain=retrain,
        nomosek=nomosek,
        minimal=minimal,
        one_iteration=one_iteration,
        switch_loss=switch_loss,
        start_script=script,
        )
        
        
    sample_list = ['01/']
    
    svm_segmenter.process_all_samples(sample_list)

    
