
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
from svm_rw_api import SVMRWMeanAPI
from svm_rw_api import MetaAnchor, MetaLaplacianFunction
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
        #switch_loss        = kwargs.pop('switch_loss', False)

        crop = kwargs.pop('crop','none')
        if crop=='none':
            self.crop = False
        else:
           self.crop = True
           ncrop = int(crop)
           self.slice_size = ncrop
           self.slice_step = ncrop

        ntrain = kwargs.pop('ntrain', 'all')
        if ntrain.isdigit():
             n = int(ntrain)
             self.select_vol = slice(n,n+1)
        else:
             self.select_vol = slice(None)
               
        ## parameters for rw learning
        self.rwparams_svm = {
            'labelset':self.labelset,
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
            }
        
        ## parameters for svm api
        self.svm_api_params = {
            'loss_type': self.loss_type, #'laplacian','squareddif', 'ideal', 'none'
            'loss_factor': self.loss_factor,
            'approx_aci': self.approx_aci,
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
            'Cprime': Cprime,
            'nitermax': 100,
            'epsilon': 1e-5,
            #'do_switch_loss': switch_loss,
            # latent
            'latent_niter_max': 100,
            'latent_epsilon': 1e-3,
            'latent_use_parallel': self.use_parallel,
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
            self.hand_tuned_w = [1.0, 0.0, 0.0, 0.0, 1e-2, 0.0, 0.0]
            self.weight_functions = {
                'std_b10'     : lambda im,i,j: wflib.weight_std(im,i,j, beta=10),
                'std_b50'     : lambda im,i,j: wflib.weight_std(im,i,j, beta=50),
                'std_b100'    : lambda im,i,j: wflib.weight_std(im,i,j, beta=100),
                'inv_b100o1'  : lambda im,i,j: wflib.weight_inv(im,i,j, beta=100, offset=1),
                }
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
            logger.info('using latent?: {}'.format(self.use_latent))
            logger.info('train data: {}'.format(ntrain))
            strkeys = ', '.join(self.laplacian_names)
            logger.info('laplacian functions (in order): {}'.format(strkeys))
            strkeys = ', '.join(self.prior_names)
            logger.info('prior models (in order): {}'.format(strkeys))
            logger.info('using loss type: {}'.format(self.loss_type))
            logger.info('SVM parameters: {}'.format(self.svmparams))
            logger.info('Computing one iteration at a time ?: {}'.format(self.one_iteration))
            if self.debug:
                logger.info('debug mode, no saving')
            else:
                logger.info('writing svm output to: {}'.format(self.dir_svm))
        

    def make_training_set(self,test, fold=None):
        if fold is None:
            fold = [test]

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
                file_seg = self.dir_reg + test + train + 'regseg.hdr'
                file_im  = self.dir_reg + test + train + 'reggray.hdr'
                
                ## load image
                im  = io_analyze.load(file_im)
                im = im/np.std(im) # normalize image by std
                
                ## load segmentation
                seg = io_analyze.load(file_seg).astype(int)
                seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]

                if self.crop:
                    ## if split training images into smaller sets
                    pmask = -1 * np.ones(seg.shape, dtype=int)
                    pmask.flat[self.prior['imask']] = np.arange(len(self.prior['imask']))
                    nslice = im.shape[0]
                    for i in range(nslice/self.slice_step):    
                        istart = i*self.slice_step
                        iend = np.minimum(nslice, i*self.slice_step + self.slice_size)
                        if istart < slice_border or istart > (im.shape[0] - slice_border):
                            continue
                        islices = np.arange(istart, iend)
                        if np.all(seg[islices]==self.labelset[0]) or \
                           np.all(self.seeds[islices]>=0):
                            continue
                        logger.debug('ivol {}, slices: start end: {} {}'.format(len(images),istart, iend))
                        bin = (seg[islices].ravel()==np.c_[self.labelset]) # make bin vector z
                        pmaski = pmask[islices]
                        imask  = np.where(pmaski.ravel()>0)[0]
                        iimask = pmaski.flat[imask]
                        #iimask = pmask[islices]
                        #iimask = iimask[iimask>=0]
                        
                        ## append to training set
                        images.append(im[islices])
                        segmentations.append(bin)
                        metadata.append({'islices': islices, 'imask':imask , 'iimask': iimask})

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
                iselect = np.arange(len(images))
                iselect = iselect[np.random.randint(
                    0,len(iselect),
                    np.minimum(nmaxvol, len(iselect)),
                    )]
                iselect = np.sort(iselect)
                logger.info('selected training: {}'.format(iselect))
                images = [images[i] for i in iselect]
                segmentations = [segmentations[i] for i in iselect]
                metadata = [metadata[i] for i in iselect]

            ntrain = len(images)
            logger.info('Learning with {} training examples'\
                .format(ntrain))
            self.training_set = (images, segmentations, metadata) 


 
    def train_svm(self,test,outdir=''):
        images, segmentations, metadata = self.training_set
        #try:
        if 1:
            import time                
            ## learn struct svm
            logger.debug('started root learning')
            wref = self.hand_tuned_w
            if self.use_latent:
                if self.one_iteration:
                    self.svmparams.pop('latent_niter_max',0) # remove kwarg
                    self.svm = latent_svm.LatentSVM(
                        self.svm_rwmean_api.compute_loss,
                        self.svm_rwmean_api.compute_psi,
                        self.svm_rwmean_api.compute_mvc,
                        self.svm_rwmean_api.compute_aci,
                        one_iteration=self.one_iteration,
                        latent_niter_max=1,
                        **self.svmparams
                        )

                    # if we're computing one iteration at a time
                    if os.path.isfile(outdir + 'niter.txt'):
                        ## continue previous work
                        niter = np.loadtxt(outdir + 'niter.txt',dtype=int)
                        ys = np.load(outdir + 'ys.npy')
                        w = np.loadtxt(outdir + 'w_{}.txt'.format(niter))
                        
                        curr_iter = niter + 1
                        logger.info('latent svm: iteration {}, with w={}'.format(curr_iter,w))
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
                        niter = 1
                        w0 = self.hand_tuned_w
                        logger.info('latent svm: first iteration. w0 = {}'.format(w0))
                        w,xi,ys,info = self.svm.train(
                            images, segmentations, metadata, w0=w0, wref=wref, **self.trainparams)
                       
                    # save output for next iteration
                    if not self.debug and not info['converged']:
                        np.savetxt(outdir + 'niter.txt', [niter], fmt='%d')
                        np.savetxt(outdir + 'w_{}.txt'.format(niter), w)
                        np.save(outdir + 'ys.npy', ys)
                        
                        #logger.warning('Exiting program. Run script again to continue.')
                        #if self.use_parallel:
                        #    for n in range(1, self.MPI_size):
                        #        self.comm.send(('stop',1, {}),dest=n)
                        logger.info('you should run command line: qsub -k oe {}'.format(self.start_script))

                else:
                    ## do all iterations
                    self.svm = latent_svm.LatentSVM(
                        self.svm_rwmean_api.compute_loss,
                        self.svm_rwmean_api.compute_psi,
                        self.svm_rwmean_api.compute_mvc,
                        self.svm_rwmean_api.compute_aci,
                        one_iteration=self.one_iteration,
                        **self.svmparams
                        )

                    logger.info('latent svm: start all iterations')
                    w0 = self.hand_tuned_w
                    w,xi,ys,info = self.svm.train(
                        images, segmentations, metadata, w0=w0, wref=wref, **self.trainparams)
            
            else:
                ## baseline: use binary ground truth with struct SVM
                self.svm = struct_svm.StructSVM(
                    self.svm_rwmean_api.compute_loss,
                    self.svm_rwmean_api.compute_psi,
                    self.svm_rwmean_api.compute_mvc,
                    **self.svmparams
                    )                  
                w0 = self.hand_tuned_w
                w,xi,info = self.svm.train( 
                    images, segmentations, metadata, 
                    w0=w0, wref=wref, **self.trainparams)

        #except Exception as e:
        else:
            import traceback
            logger.error('{}: {}'.format(e.message, e.__class__.__name__))
            traceback.print_exc()
        #finally:
        if 1:
            ##kill signal
            if self.use_parallel:
                logger.info('root finished training svm on {}. about to kill workers'\
                    .format(test))
                for n in range(1, self.MPI_size):
                    logger.debug('sending kill signal to worker #{}'.format(n))
                    self.comm.send(('stop',None,{}),dest=n)
            return w,xi
            #logger.debug('worker #{} about to exit'.format(rank))
  
        
    def run_svm_inference(self,test,w, test_dir):
        logger.info('running inference on: {}'.format(test))
        
        ## normalize w
        # w = w / np.sqrt(np.dot(w,w))
        strw = ' '.join('{:.3}'.format(val) for val in np.asarray(w)*self.psi_scale)
        logger.debug('scaled w=[{}]'.format(strw))
    
        weights_laplacians = np.asarray(w)[self.indices_laplacians]
        weights_laplacians_h = np.asarray(self.hand_tuned_w)[self.indices_laplacians]
        weights_priors = np.asarray(w)[self.indices_priors]
        weights_priors_h = np.asarray(self.hand_tuned_w)[self.indices_priors]
    
        ## segment test image with trained w
        '''
        def meta_weight_functions(im,i,j,_w):    
            data = 0
            for iwf,wf in enumerate(self.laplacian_functions):
                _data = wf(im,i,j)
                data += _w[iwf]*_data
            return data
        weight_function = lambda im: meta_weight_functions(im,i,j,weights_laplacians)
        weight_function_h = lambda im: meta_weight_functions(im,i,j,weights_laplacians_h)
        '''
        weight_function = MetaLaplacianFunction(
            weights_laplacians,
            self.laplacian_functions)
        
        weight_function_h = MetaLaplacianFunction(
            weights_laplacians_h,
            self.laplacian_functions)
        
        ## load images and ground truth
        file_seg = self.dir_reg + test + 'seg.hdr'
        file_im  = self.dir_reg + test + 'gray.hdr'
        im  = io_analyze.load(file_im)
        seg = io_analyze.load(file_seg)
        seg.flat[~np.in1d(seg.ravel(),self.labelset)] = self.labelset[0]
        
        nim = im/np.std(im) # normalize image by std

        ## test training data ?
        inference_train = True
        if inference_train:
            train_ims, train_segs, train_metas = self.training_set
            for tim, tz, tmeta in zip(train_ims, train_segs, train_metas):
                ## retrieve metadata
                islices = tmeta.pop('islices',None)
                imask = tmeta.pop('imask', None)
                iimask = tmeta.pop('iimask',None)
                if islices is not None:
                    tseeds = self.seeds[islices]
                    tprior = {
                        'data': np.asarray(self.prior['data'])[:,iimask],
                        'imask': imask,
                        'variance': np.asarray(self.prior['variance'])[:,iimask],
                        'labelset': self.labelset,
                        }
                    if 'intensity' in self.prior: 
                        tprior['intensity'] = self.prior['intensity']
                else:
                    tseeds = self.seeds
                    tprior = self.prior

                ## prior
                tseg = self.labelset[np.argmax(tz, axis=0)].reshape(tim.shape)
                tanchor_api = MetaAnchor(
                    tprior,
                    self.prior_functions,
                    weights_priors,
                    image=tim,
                    )
                tsol,ty = rwsegment.segment(
                    tim, 
                    tanchor_api, 
                    seeds=tseeds,
                    weight_function=weight_function,
                    **self.rwparams_inf
                    )
                ## compute Dice coefficient
                tdice = compute_dice_coef(tsol, tseg, labelset=self.labelset)
                logger.info('Dice coefficients for train: \n{}'.format(tdice))
                nlabel = len(self.labelset)
                tflatmask = np.zeros(ty.shape, dtype=bool)
                tflatmask[:,imask] = True
                loss0 = loss_functions.ideal_loss(tz,ty,mask=tflatmask)
                logger.info('Tloss = {}'.format(loss0))
                ## loss2: squared difference with ztilde
                loss1 = loss_functions.anchor_loss(tz,ty,mask=tflatmask)
                logger.info('SDloss = {}'.format(loss1))
                ## loss3: laplacian loss
                loss2 = loss_functions.laplacian_loss(tz,ty,mask=tflatmask)
                logger.info('LAPloss = {}'.format(loss2))


                tanchor_api_h = MetaAnchor(
                    tprior,
                    self.prior_functions,
                    weights_priors_h,
                    image=tim,
                    )
            
                tsol,ty = rwsegment.segment(
                    tim, 
                    tanchor_api_h, 
                    seeds=tseeds,
                    weight_function=weight_function_h,
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
                loss2 = loss_functions.laplacian_loss(tz,ty,mask=tflatmask)
                logger.info('LAPloss (hand-tuned) = {}'.format(loss2))
                break
 
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
        logger.info('Dice coefficients: \n{}'.format(dice))

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
            outdir = self.dir_inf + test_dir
            logger.info('saving data in: {}'.format(outdir))
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
                
            #io_analyze.save(outdir + 'im.hdr',im.astype(np.int32))
            #np.save(outdir + 'y.npy',y)        
            #io_analyze.save(outdir + 'sol.hdr',sol.astype(np.int32))
            np.savetxt(outdir + 'objective.txt', [obj])
            np.savetxt(
                outdir + 'dice.txt', 
                np.c_[dice.keys(),dice.values()],fmt='%d %f')
        
            f = open(outdir + 'losses.txt', 'w')
            f.write('ideal_loss\t{}\n'.format(loss0))
            f.write('anchor_loss\t{}\n'.format(loss1))
            f.write('laplacian_loss\t{}\n'.format(loss2))
            f.close()
        
    def process_sample(self, test, fold=None):
        if fold is not None:
            test_dir = 'f{}_{}'.format(fold[0][:2], test)
        else:
            test_dir = test
 
        if self.isroot:
            prior, mask = load_or_compute_prior_and_mask(
                test,force_recompute=self.force_recompute_prior, fold=fold)
            
            if self.use_parallel:
                # have only the root process compute the prior 
                # and pass it to the other processes
                self.comm.bcast((dict(prior.items()),mask),root=0)    
        else:
            prior,mask = self.comm.bcast(None,root=0)
        
        self.prior = prior
        self.seeds = (-1)*mask.astype(int)
        
        ## training set
        self.make_training_set(test, fold=fold)

        ## training
        if self.retrain:
            outdir = self.dir_svm + test_dir
            if not self.debug and not os.path.isdir(outdir):
                os.makedirs(outdir)

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
                w,xi = self.train_svm(test,outdir=outdir)
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
            if self.isroot and not self.retrain:    
                outdir = self.dir_svm + test
                logger.warning('Not retraining svm')
                w = np.loadtxt(outdir + 'w')
        
        ## inference
        if self.isroot: 
            self.w = w
            self.run_svm_inference(test,w, test_dir=test_dir)
        
        
    
    def process_all_samples(self,sample_list,fold=None):
        for test in sample_list:
            if self.isroot:
                logger.info('--------------------------')
                logger.info('test data: {}'.format(test))
            self.process_sample(test, fold)
        #if self.isroot:
        #   for n in range(1, self.MPI_size):
        #       logger.debug('sending kill signal to worker #{}'.format(n))
        #       self.comm.send(('stop',None,{}),dest=n)
    
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
        )
        
        
    #sample_list = ['01/']
    for fold in config.folds:
        svm_segmenter.process_all_samples(fold, fold=fold) 
    sys.exit(1)
    
