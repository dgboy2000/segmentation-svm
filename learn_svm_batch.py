
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
reload(rwsegment)
from rwsegment import rwsegment_prior_models as models
reload(models)
from rwsegment import struct_svm
reload(struct_svm)
from rwsegment import latent_svm
reload(latent_svm)
from rwsegment import svm_worker
reload(svm_worker)
from rwsegment import loss_functions
reload(loss_functions)
from rwsegment.rwsegment import BaseAnchorAPI
from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
import svm_rw_api
reload(svm_rw_api)
import svm_rw_functions
reload(svm_rw_functions)
from svm_rw_api import SVMRWMeanAPI
from rwsegment import mpi

from rwsegment import utils_logging
logger = utils_logging.get_logger('learn_svm_batch',utils_logging.DEBUG)

class SVMSegmenter(object):

    def __init__(self,
            use_parallel=True, 
            **kwargs
            ):
    
        ## paths
        self.dir_reg = config.dir_reg
        self.dir_inf = config.dir_inf
        self.dir_svm = config.dir_svm
        self.training_vols = config.vols

        ## params
        self.force_recompute_prior = False
        self.use_parallel = use_parallel    

        self.labelset = np.asarray(config.labelset)

        C = kwargs.pop('C',1.0)
        Cprime = kwargs.pop('Cprime',0.0)
        self.scale_only    = kwargs.pop('scale_only', False)
        self.loss_type     = kwargs.pop('loss_type', 'squareddiff')
        self.loss_factor   = kwargs.pop('loss_factor', 1.)
        self.use_latent    = kwargs.pop('use_latent', False)
        self.approx_aci    = kwargs.pop('approx_aci', False)
        self.debug         = kwargs.pop('debug', False)
        self.retrain       = kwargs.pop('retrain', True)
        self.minimal_svm   = kwargs.pop('minimal', False)
        self.one_iteration = kwargs.pop('one_iteration', False)
        self.start_script  = kwargs.pop('start_script', '')       
        self.use_mosek     = kwargs.pop('use_mosek',True)

        crop = kwargs.pop('crop','none')
        if crop=='none':
            self.crop = False
        else:
           self.crop = True
           ncrop = int(crop)
           self.slice_size = ncrop
           self.slice_step = ncrop

        ntrain = kwargs.pop('ntrain', 'all')
        if ntrain in ['all']:
             self.select_vol = slice(None)
        elif ntrain.isdigit():
             n = int(ntrain)
             self.select_vol = slice(n,n+1)
               
        ## parameters for rw learning
        self.rwparams_svm = {
            # optimization
            'rtol': 1e-6,
            'maxiter': 1e3,
            'per_label':True,
            'optim_solver':'unconstrained',
            # contrained optim
            'use_mosek': self.use_mosek,
            'logbarrier_mu': 10,
            'logbarrier_initial_t': 10,
            'logbarrier_modified': False,
            'logbarrier_maxiter': 10,
            'newton_maxiter': 50,
            ## duald
            'duald_gamma': kwargs.pop('duald_gamma'),
            'duald_niter': kwargs.pop('duald_niter'),
            'duald_epsilon': 1e-3, 
            'duald_size_sub': (2,10,10),
            }
        
        ## parameters for svm api
        self.svm_api_params = {
            'loss_type': self.loss_type, #'laplacian','squareddif', 'ideal', 'none'
            'loss_factor': self.loss_factor,
            'approx_aci': self.approx_aci,
            }
            
        ## parameters for rw inference
        self.rwparams_inf = {
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
            'Cprime': Cprime,
            'nitermax': 100,
            'epsilon': 1e-5,
            # latent
            'latent_niter_max': 100,
            'latent_epsilon': 1e-3,
            'latent_use_parallel': self.use_parallel & self.approx_aci,
            }

        self.trainparams = {
            'scale_only': self.scale_only,
            }       
     
        ## weight functions
        if self.minimal_svm:
            self.hand_tuned_w = [1, 1e-2]
            self.weight_functions = {'std_b50': lambda im: wflib.weight_std(im, beta=50)}
            self.prior_models = {'constant': models.Constant}
        else:
            self.laplacian_functions = [
                {'name':'std_b10',  'func': lambda im,i,j: wflib.weight_std(im,i,j, beta=10),  'default': 0.0},
                {'name':'std_b50',  'func': lambda im,i,j: wflib.weight_std(im,i,j, beta=50),  'default': 1.0},
                {'name':'std_b100', 'func': lambda im,i,j: wflib.weight_std(im,i,j, beta=100), 'default': 0.0},
                {'name':'inv_b100o1', 'func': lambda im,i,j: wflib.weight_inv(im,i,j, beta=100, offset=1), 'default': 0.0},
                ]
            self.prior_models = [
                {'name': 'constant',  'default': 0.0},
                {'name': 'entropy',   'default': 1e0}, #1e-2 is better
                {'name': 'intensity', 'default': 0.0},
                ]

        ## parallel ?
        if self.use_parallel:
            self.comm = mpi.COMM
            self.MPI_rank = mpi.RANK
            self.MPI_size = mpi.SIZE
            self.isroot = self.MPI_rank==0
            if self.MPI_size==1:
                logger.warning('Found only one process. Not using parallel')
                self.use_parallel = False
            self.svmparams['use_parallel'] = self.use_parallel
            #self.svmparams['latent_use_parallel'] = self.use_parallel
        else:
            self.isroot = True
            
        if self.isroot:
            logger.info('passed these command line arguments: {}'.format(str(sys.argv)))
            logger.info('using parallel?: {}'.format(use_parallel))
            logger.info('using latent?: {}'.format(self.use_latent))
            logger.info('train data: {}'.format(ntrain))
            logger.info('laplacian functions (in order): {}'.format(
                ', '.join([lap['name'] for lap in self.laplacian_functions])))
            logger.info('prior models (in order): {}'.format(
                ', '.join([mod['name'] for mod in self.prior_models])))
            logger.info('using loss type: {}'.format(self.loss_type))
            logger.info('SVM parameters: {}'.format(self.svmparams))
            logger.info('Computing one iteration at a time ?: {}'.format(self.one_iteration))
            if self.debug:
                logger.info('debug mode, no saving')
            else:
                logger.info('writing svm output to: {}'.format(self.dir_svm))

    def make_training_set(self, fold, seeds):
    
        ## training images and segmentations
        if self.isroot:
            slice_border = 20 # do not consider top and bottom slices
            images = []
            segmentations = []
            metadata = []

            for train in self.training_vols:
                if train in fold: continue
                logger.info('loading training data: {}'.format(train))
                 
                ## file names
                file_seg = self.dir_reg + fold[0] + train + 'regseg.hdr'
                file_im  = self.dir_reg + fold[0] + train + 'reggray.hdr'
                
                ## load image
                im  = io_analyze.load(file_im)
                im = im/np.std(im) # normalize image by std
                
                ## load segmentation
                seg = io_analyze.load(file_seg).astype(int)
                seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]

                if self.crop:
                    ## if split training images into smaller sets
                    nslice = im.shape[0]
                    for i in range(nslice/self.slice_step):    
                        istart = i*self.slice_step
                        iend = np.minimum(nslice, i*self.slice_step + self.slice_size)
                        if istart < slice_border or istart > (im.shape[0] - slice_border):
                            continue
                        islices = np.arange(istart, iend)
                        if np.all(seg[islices]==self.labelset[0]) or \
                           np.all(seeds[islices]>=0):
                            continue
                        #logger.debug('ivol {}, slices: start end: {} {}'.format(len(images),istart, iend))
                        bin = (seg[islices].ravel()==np.c_[self.labelset]) # make bin vector z
                        
                        ## append to training set
                        images.append(im[islices])
                        segmentations.append(bin)
                        metadata.append({'islices': islices, 'shape': im.shape})

                        ## break loop
                        if len(images) == self.select_vol.stop:
                            break 

                else:
                    bin = (seg.ravel()==np.c_[self.labelset])# make bin vector z                
                    ## append to training set
                    images.append(im)
                    segmentations.append(bin)
                    metadata.append({})

                ## break loop
                if len(images) == self.select_vol.stop:
                    break 

            nmaxvol = 100
            if len(images) > nmaxvol:
                randomizer = np.random.RandomState(seed=0)
                iselect = randomizer.permutation(np.arange(len(images)))[:nmaxvol] 
                iselect = np.sort(iselect)
                logger.info('selected training: {}'.format(iselect))
                images = [images[i] for i in iselect]
                segmentations = [segmentations[i] for i in iselect]
                metadata = [metadata[i] for i in iselect]
		    

            ntrain = len(images)
            logger.info('Learning with {} training examples'\
                .format(ntrain))
            self.training_set = (
			    images[self.select_vol], 
				segmentations[self.select_vol], 
				metadata[self.select_vol]) 

 
    def train_svm(self, fold, outdir=''):
        api = self.svm_rwmean_api
        seeds = self.seeds
        
        ## training set
        self.make_training_set(fold, seeds)
        images, segmentations, metadata = self.training_set
        
        if 1:
            import time                
            ## learn struct svm
            logger.debug('started root learning')
            nlabel = len(self.labelset)
            wref = [f['default'] for f in self.laplacian_functions] + \
                   [m['default'] for m in self.prior_models ]
            w0 = wref
            wbad = [0.,0.,0.,1.,0.,0.,1.] ## rubbish init for struct  svm
            if self.use_latent:
                if self.one_iteration:
                    self.svmparams.pop('latent_niter_max',0) # remove kwarg
                    self.svm = latent_svm.LatentSVM(
                        api.compute_loss,
                        api.compute_psi,
                        api.compute_mvc,
                        api.compute_aci,
                        one_iteration=self.one_iteration,
                        latent_niter_max=1,
                        winit=wbad,
                        **self.svmparams
                        )

                    # if we're computing one iteration at a time
                    if os.path.isfile(outdir + 'niter.txt'):
                        ## continue previous work
                        niter = np.loadtxt(outdir + 'niter.txt',dtype=int)
                        if not os.path.exists('ys_{}.npz'.format(niter)):
                            ys = np.load(outdir + 'ys_{}.npy'.format(niter))
                        else:
                            ys = np.load(outdir + 'ys_{}.npz'.format(niter))['ys']
                           
                        w = np.loadtxt(outdir + 'w_{}.txt'.format(niter)).tolist()
                        
                        niter = niter + 1
                        logger.info('latent svm: iteration {}, with w={}'.format(niter,w))
                        w,xi,ys,info = self.svm.train(
                            images, 
                            segmentations, 
                            metadata, 
                            w0=w,
                            wref=wref,
                            init_latents=ys, 
                            **self.trainparams)
                    else:
                        ## start learning
                        niter = 0
                        logger.info('latent svm: first iteration. w0 = {}'.format(w0))
                        w,xi,ys,info = self.svm.train(
                            images, segmentations, metadata, w0=w0, wref=wref, **self.trainparams)
                       
                    # save output for next iteration
                    if not self.debug:
                        np.savetxt(outdir + 'niter.txt', [niter], fmt='%d')
                        np.savetxt(outdir + 'w_{}.txt'.format(niter), w)
                        #np.save(outdir + 'ys_{}.npy'.format(niter), ys)
                        np.savez_compressed(outdir + 'ys_{}.npz'.format(niter), {'ys':ys})
                        
                        #logger.warning('Exiting program. Run script again to continue.')
                        #if self.use_parallel:
                        #    for n in range(1, self.MPI_size):
                        #        self.comm.send(('stop',1, {}),dest=n)
                    if not info['converged']:
                        logger.info('you should run command line: qsub -k oe {}'.format(self.start_script))

                else:
                    ## do all iterations
                    self.svm = latent_svm.LatentSVM(
                        api.compute_loss,
                        api.compute_psi,
                        api.compute_mvc,
                        api.compute_aci,
                        one_iteration=self.one_iteration,
                        winit=wbad,
                        **self.svmparams
                        )

                    logger.info('latent svm: start all iterations')
                    w,xi,ys,info = self.svm.train(
                        images, segmentations, metadata, w0=w0, wref=wref, **self.trainparams)
            
            else:
                ## baseline: use binary ground truth with struct SVM
                self.svm = struct_svm.StructSVM(
                    api.compute_loss,
                    api.compute_psi,
                    api.compute_mvc,
                    winit=wbad,
                    **self.svmparams
                    )
                w,xi,info = self.svm.train( 
                    images, segmentations, metadata, 
                    w0=w0, wref=wref, **self.trainparams)

        #except Exception as e:
        else:
            import traceback
            logger.error('{}: {}'.format(e.message, e.__class__.__name__))
            traceback.print_exc()
            #import ipdb; ipdb.set_trace()
            
        #finally:
        if 1:

            return w,xi
            #logger.debug('worker #{} about to exit'.format(rank))
  
        
    def run_svm_inference(self, fold, w):
        ## w
        strw = ' '.join('{:.3}'.format(val) for val in np.asarray(w))
        logger.debug('w=[{}]'.format(strw))    
    
        #import ipdb; ipdb.set_trace()
    
        ## laplacian functions
        nlabel = len(self.labelset)
        nlaplacian = len(self.laplacian_functions)
        lweights = w[:nlaplacian]
        laplacian_function = svm_rw_functions.LaplacianWeights(
            nlabel,self.laplacian_functions, weights=lweights) 
        laplacian_function_h = svm_rw_functions.LaplacianWeights(
            nlabel,self.laplacian_functions) 
        
        ## anchor weights        
        aweights = w[nlaplacian:]
        api, prior_models, seeds = self.get_prior_and_seeds(fold[0], fold)
        
        inference_train = True
        if inference_train and hasattr(self, "training_set"):
            train_ims, train_segs, train_metas = self.training_set
            for tim, tz, tmeta in zip(train_ims, train_segs, train_metas): 
                islices = tmeta.pop('islices', slice(None))
                shape = tmeta.pop('shape', tim.shape)
                tseg = self.labelset[np.argmax(tz,axis=0)].reshape(tim.shape)
                tseeds = seeds[islices]
                tflatmask = (tseeds<0).ravel()*np.ones((len(self.labelset),1))

                ## anchors
                amodels  = svm_rw_functions.reslice_models(shape, prior_models, islices=islices)
                anchor_api = svm_rw_functions.MetaAnchorApi(amodels, weights=aweights)
                
                ## segment
                tsol,ty = rwsegment.segment(
                    tim, 
                    anchor_api, 
                    self.labelset,
                    seeds=tseeds,
                    laplacian_function=laplacian_function,
                    **self.rwparams_inf
                    )

                ## compute Dice coefficient
                tdice = compute_dice_coef(tsol, tseg, labelset=self.labelset)
                logger.info('Dice coefficients for train: \n{}'.format(tdice))
                loss0 = loss_functions.ideal_loss(tz,ty,mask=tflatmask)
                logger.info('Tloss = {}'.format(loss0))
                ## loss2: squared difference with ztilde
                loss1 = loss_functions.anchor_loss(tz,ty,mask=tflatmask)
                logger.info('SDloss = {}'.format(loss1))
                ## loss3: laplacian loss
                loss2 = loss_functions.laplacian_loss(tz,ty,mask=tflatmask)
                logger.info('LAPloss = {}'.format(loss2))

                ## hand tuned parameters
                anchor_api_h = svm_rw_functions.MetaAnchorApi(amodels)

                tsol,ty = rwsegment.segment(
                    tim, 
                    anchor_api_h,
                    self.labelset,
                    seeds=tseeds,
                    laplacian_function=laplacian_function_h,
                    **self.rwparams_inf
                    )
                ## compute Dice coefficient
                tdice = compute_dice_coef(tsol, tseg, labelset=self.labelset)
                logger.info('Dice coefficients for train (hand-tuned): \n{}'.format(tdice))
                loss0 = loss_functions.ideal_loss(tz,ty,mask=tflatmask)
                logger.info('Tloss (hand-tuned) = {}'.format(loss0))
                ## loss2: squared difference with ztilde
                loss1 = loss_functions.anchor_loss(tz,ty,mask=tflatmask)
                logger.info('SDloss (hand-tuned) = {}'.format(loss1))
                ## loss3: laplacian loss
                # loss2 = loss_functions.laplacian_loss(tz,ty,mask=tflatmask)
                # logger.info('LAPloss (hand-tuned) = {}'.format(loss2))
                break
   
   
        for test in fold:
            logger.info('running inference on: {}'.format(fold))
            ## load images and ground truth
            file_seg = self.dir_reg + test + 'seg.hdr'
            file_im  = self.dir_reg + test + 'gray.hdr'
            im  = io_analyze.load(file_im)
            seg = io_analyze.load(file_seg)
            seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]
            nim = im/np.std(im) # normalize image by std

            ## prior
            api, prior_models, seeds = self.get_prior_and_seeds(test, fold)
            anchor_api = svm_rw_functions.MetaAnchorApi(prior_models, weights=aweights)
        
            sol,y = rwsegment.segment(
                nim, 
                anchor_api, 
                self.labelset,
                seeds=seeds,
                laplacian_function=laplacian_function,
                **self.rwparams_inf
                )
            
            ## compute Dice coefficient
            dice = compute_dice_coef(sol, seg,labelset=self.labelset)
            logger.info('Dice coefficients: \n{}'.format(dice))

            ## objective
            en_rw = rwsegment.energy_rw(
                nim, y, self.labelset,
                seeds=seeds,laplacian_function=laplacian_function, **self.rwparams_inf)
            en_anchor = rwsegment.energy_anchor(
                nim, y, anchor_api, self.labelset,
                seeds=seeds, **self.rwparams_inf)
            obj = en_rw + en_anchor
            logger.info('Objective = {:.3}'.format(obj))

            
            ## compute losses
            z = seg.ravel()==np.c_[self.labelset]
            mask = seeds < 0
            flatmask = mask.ravel()*np.ones((len(self.labelset),1))
            
            ## loss 0 : 1 - Dice(y,z)
            loss0 = loss_functions.ideal_loss(z,y,mask=flatmask)
            logger.info('Tloss = {}'.format(loss0))
            
            ## loss2: squared difference with ztilde
            loss1 = loss_functions.anchor_loss(z,y,mask=flatmask)
            logger.info('SDloss = {}'.format(loss1))
            
            ## loss3: laplacian loss
            # loss2 = loss_functions.laplacian_loss(z,y,mask=flatmask)
            # logger.info('LAPloss = {}'.format(loss2))

            ## loss4: linear loss
            loss3 = loss_functions.linear_loss(z,y,mask=flatmask)
            logger.info('LINloss = {}'.format(loss3))
           
            ## saving
            if self.debug:
                pass
            elif self.isroot:
                fold_dir = 'f{}_{}/'.format(fold[0][:2], test)
                outdir = self.dir_inf + fold_dir
                
                logger.info('saving data in: {}'.format(outdir))
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                    
                np.savetxt(outdir + 'objective.txt', [obj])
                np.savetxt(
                    outdir + 'dice.txt', 
                    np.c_[dice.keys(),dice.values()],fmt='%d %f')
            
                f = open(outdir + 'losses.txt', 'w')
                f.write('ideal_loss\t{}\n'.format(loss0))
                f.write('anchor_loss\t{}\n'.format(loss1))
                # f.write('laplacian_loss\t{}\n'.format(loss2))
                f.close()
        
        
    def get_prior_and_seeds(self, test, fold, bcast=False):
        if self.isroot:
            prior, mask = load_or_compute_prior_and_mask(
                test, 
                force_recompute=self.force_recompute_prior, 
                fold=fold)
            
            if bcast and self.use_parallel:
                # have only the root process compute the prior 
                # and pass it to the other processes
                self.comm.bcast((dict(prior.items()),mask),root=0)    
        elif bcast:
            prior,mask = self.comm.bcast(None,root=0)
        
        ## instantiate models
        imask = prior['imask']
        average = prior['data']
        variance = prior['variance']
        im_avg, im_var = prior['intensity']
        
        prior_models = [dict(p) for p in self.prior_models]
        prior_models[0]['api'] = models.Constant(imask, average)
        prior_models[1]['api'] = models.Entropy_no_D(imask, average)
        prior_models[2]['api'] = models.Intensity(im_avg, im_var)       
        
        ## seeds
        seeds = (-1)*mask.astype(int)
        
        api = SVMRWMeanAPI(
            self.laplacian_functions, 
            prior_models,
            seeds,
            self.labelset,
            self.rwparams_svm,
            **self.svm_api_params)
        
        return api, prior_models, seeds
        
        
    def process_sample(self, fold):
        fold_dir = 'f{}/'.format(fold[0][:2])
        outdir = self.dir_svm + fold_dir

        api, prior_models, seeds = self.get_prior_and_seeds(fold[0], fold, bcast=True)
        self.svm_rwmean_api = api
        self.seeds = seeds
        
        ## training
        if self.retrain:
            
            if self.isroot:
                if not self.debug and not os.path.exists(outdir):
                    os.makedirs(outdir)

                w,xi = self.train_svm(fold, outdir=outdir)
                if self.debug:
                    pass
                elif self.isroot:
                    np.savetxt(outdir + 'w',w)
                    np.savetxt(outdir + 'xi',[xi])     
            else:
                ## parallel 
                rank = self.MPI_rank
                logger.debug('started worker #{}'.format(rank))                
                worker = svm_worker.SVMWorker(self.svm_rwmean_api)
                worker.work()

        else:
            if self.isroot:
                logger.warning('Not retraining svm')
                w = np.loadtxt(outdir + 'w')
        
        ## inference
        if self.isroot:
            self.w = w
            self.run_svm_inference(fold,w)
        
        ##kill signal
        if self.isroot and self.use_parallel:
            logger.info('root finished training svm on {}. about to kill workers'\
                .format(fold))
            for n in range(1, self.MPI_size):
                logger.debug('sending kill signal to worker #{}'.format(n))
                self.comm.send(('stop',None,{}),dest=n)
        
    
    def process_all_samples(self,fold):
        logger.info('--------------------------')
        logger.info('test data: {}'.format(fold))
        self.process_sample(fold)
    
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
        '--loss', dest='loss', 
        default='squareddiff', type=str,
        help='loss type ("squareddiff", "laplacian", "none", "ideal")',
        )
    opt.add_option( # loss factor
        '--loss_factor', dest='loss_factor', 
        default=1, type=float,
        help='',
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
        '--use_mosek', dest='use_mosek', 
        default='True', type=str,
        help='use mosek in constrained optim ?',
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
    opt.add_option( # Cprime
        '--Cprime', dest='Cprime', 
        default=0.0, type=float,
        help='Cprime value',
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
    opt.add_option(
        '--crop', dest='crop', 
        default='none', type=str,
        help='crop images (integer or none)',
        )  
    opt.add_option(
        '--approx_aci', dest='approx_aci', 
        default=False, action="store_true",
        help='use approximate inference',
        )  
    opt.add_option(
        '--scale_only', dest='scale_only', 
        default=False, action="store_true",
        help='',
        )     
    opt.add_option(
        '--duald_niter', dest='duald_niter', 
        default=10, type=int,
        help='',
        )     
    opt.add_option(
        '--duald_gamma', dest='duald_gamma', 
        default=1e2, type=float,
        help='',
        )
    opt.add_option(
        '--fold', dest='fold', 
        default=0, type=int,
        help='',
        )

    (options, args) = opt.parse_args()

    use_parallel = bool(options.parallel)
    ntrain = options.ntrain
    debug = options.debug
    retrain = 1 - options.noretrain
    minimal = options.minimal
    one_iteration = options.one_iter
    switch_loss = options.switch_loss
    script = options.script

    folder = options.folder #unused

    ''' start script '''
    svm_segmenter = SVMSegmenter(
        C=options.C,
        Cprime=options.Cprime,
        use_parallel=use_parallel,
        use_latent=options.latent,
        loss_type=options.loss,
        loss_factor=options.loss_factor,
        ntrain=ntrain,
        debug=debug,
        retrain=retrain,
        minimal=minimal,
        one_iteration=one_iteration,
        switch_loss=switch_loss,
        start_script=script,
        crop=options.crop,
        approx_aci=options.approx_aci,
        use_mosek=(options.use_mosek in ['True', 'true', '1']),
        scale_only=options.scale_only,
        duald_gamma=options.duald_gamma,
        duald_niter=options.duald_niter,
        )
        
    
    if options.fold < 0:
        folds = config.folds
    else:
        folds = config.folds[options.fold:options.fold+1]
    for fold in folds:
       svm_segmenter.process_all_samples(fold) 
    sys.exit(1)
    
