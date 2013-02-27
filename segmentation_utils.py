import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment_prior
from rwsegment import rwsegment_pca_prior
from rwsegment import loss_functions
reload(rwsegment_prior)
reload(rwsegment_pca_prior)
reload(loss_functions)

import config
reload(config)

from rwsegment import utils_logging
logger = utils_logging.get_logger('segmentation_utils',utils_logging.DEBUG)


def compute_losses(z,y,mask):
    ## loss 0 : 1 - Dice(y,z)
    loss0 = loss_functions.ideal_loss(z,y,mask=mask)
    logger.info('Tloss = {}'.format(loss0))
    
    ## loss2: squared difference with ztilde
    loss1 = loss_functions.anchor_loss(z,y,mask=mask)
    logger.info('SDloss = {}'.format(loss1))
    
    ## loss3: laplacian loss
    loss2 = loss_functions.laplacian_loss(z,y,mask=mask)
    logger.info('LAPloss = {}'.format(loss2))

    ## loss4: linear loss
    loss3 = loss_functions.linear_loss(z,y,mask=mask)
    logger.info('LINloss = {}'.format(loss3))
    
    return loss0, loss1, loss2, loss3
    
def compute_features(test, train, y):
    im = io_analyze.load(config.dir_reg + test + train + 'reggray.hdr')
    nim = im/np.std(im)
     
    prior,mask = load_or_compute_prior_and_mask(
        test, force_recompute=False)
    seeds = (-1)*mask.astype(int)

    from rwsegment import rwsegment_prior_models as models
    from rwsegment import weight_functions as wflib
    rwparams = {
            'labelset': np.asarray(config.labelset),
        }

    weight_functions = {
        'std_b10'     : lambda im: wflib.weight_std(im, beta=10),
        'std_b50'     : lambda im: wflib.weight_std(im, beta=50),
        'std_b100'    : lambda im: wflib.weight_std(im, beta=100),
        'inv_b100o1'  : lambda im: wflib.weight_inv(im, beta=100, offset=1),
        # 'pdiff_r1b10': lambda im: wflib.weight_patch_diff(im, r0=1, beta=10),
        # 'pdiff_r2b10': lambda im: wflib.weight_patch_diff(im, r0=2, beta=10),
        # 'pdiff_r1b50' : lambda im: wflib.weight_patch_diff(im, r0=1, beta=50),
        }

    prior_models = {
        'constant': models.Constant,
        'entropy': models.Entropy_no_D,
        'intensity': models.Intensity,
        }
    
    ## indices of w
    nlaplacian = len(weight_functions)
    nprior = len(prior_models)
    indices_laplacians = np.arange(nlaplacian)
    indices_priors = np.arange(nlaplacian,nlaplacian + nprior)
  
    laplacian_functions = weight_functions.values()
    laplacian_names     = weight_functions.keys()
    prior_functions     = prior_models.values()
    prior_names         = prior_models.keys()
    

    #from svm_rw_api import MetaAnchor 
    #anchor_api = MetaAnchor(
    #    prior,
    #    prior_functions,
    #    weights_priors,
    #    image=im,
    #    )

    from rwsegment import rwsegment
    for fname in weight_functions:
        wf = weight_functions[fname]
        en_rw = rwsegment.energy_rw(
            nim,
            y,
            seeds=seeds,
            weight_function=wf,
            **rwparams
            )
        print fname, en_rw

    for mname in prior_models:
        pm = prior_models[mname](prior,1., image=nim)
        en_anchor = rwsegment.energy_anchor(
            nim,
            y,
            pm,
            seeds=seeds,
            **rwparams
            )
        print mname, en_anchor


   

def compute_objective(test, y, w):
    im = io_analyze.load(config.dir_reg + test + 'gray.hdr')
    nim = im/np.std(im)
     
    prior,mask = load_or_compute_prior_and_mask(
        test, force_recompute=False)
    seeds = (-1)*mask.astype(int)

    from rwsegment import rwsegment_prior_models as models
    from rwsegment import weight_functions as wflib
    rwparams = {
            'labelset': np.asarray(config.labelset),

            # optimization
            'rtol': 1e-6,
            'maxiter': 1e3,
            'per_label':True,
            'optim_solver':'unconstrained',
            }

    weight_functions = {
        'std_b10'     : lambda im: wflib.weight_std(im, beta=10),
        'std_b50'     : lambda im: wflib.weight_std(im, beta=50),
        'std_b100'    : lambda im: wflib.weight_std(im, beta=100),
        'inv_b100o1'  : lambda im: wflib.weight_inv(im, beta=100, offset=1),
        # 'pdiff_r1b10': lambda im: wflib.weight_patch_diff(im, r0=1, beta=10),
        # 'pdiff_r2b10': lambda im: wflib.weight_patch_diff(im, r0=2, beta=10),
        # 'pdiff_r1b50' : lambda im: wflib.weight_patch_diff(im, r0=1, beta=50),
        }

    prior_models = {
        'constant': models.Constant,
        'entropy': models.Entropy_no_D,
        'intensity': models.Intensity,
        }
    
    ## indices of w
    nlaplacian = len(weight_functions)
    nprior = len(prior_models)
    indices_laplacians = np.arange(nlaplacian)
    indices_priors = np.arange(nlaplacian,nlaplacian + nprior)
  
    laplacian_functions = weight_functions.values()
    laplacian_names     = weight_functions.keys()
    prior_functions     = prior_models.values()
    prior_names         = prior_models.keys()
    
    weights_laplacians = np.asarray(w)[indices_laplacians]
    weights_priors = np.asarray(w)[indices_priors]

    def meta_weight_functions(im,_w):
        ''' meta weight function'''
        data = 0
        for iwf,wf in enumerate(laplacian_functions):
            ij,_data = wf(im)
            data += _w[iwf]*_data
        return ij, data
    weight_function = lambda im: meta_weight_functions(im, weights_laplacians)
    
    

    from svm_rw_api import MetaAnchor 
    anchor_api = MetaAnchor(
        prior,
        prior_functions,
        weights_priors,
        image=nim,
        )

    from rwsegment import rwsegment
    en_rw = rwsegment.energy_rw(
            nim,
            y,
            seeds=seeds,
            weight_function=weight_function,
            **rwparams
            )

    en_anchor = rwsegment.energy_anchor(
            nim,
            y,
            anchor_api,
            seeds=seeds,
            **rwparams
            )
    obj = en_rw + en_anchor
    return obj

    
def load_or_compute_prior_and_mask(test, force_recompute=False, pca=False, fold=None):
    if fold is not None:
        test_name = 'f{}_{}/'.format(fold[0][:2], test)
    else:
        test_name = test
        fold = [test]
     
    labelset = np.asarray(config.labelset)
    if pca:
        outdir = config.dir_pca_prior + test_name
    else:
        outdir = config.dir_prior + test_name
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    ## load mask and prior
    prior = None
    file_mask  = outdir + 'mask.hdr'
    file_prior = outdir + 'prior.npz'
    file_segprior = outdir + 'segprior.hdr'
    file_entropymap = outdir + 'entropymap.hdr'
   
    if (not force_recompute) and os.path.exists(outdir + '/data.npz') and os.path.exists(file_mask):
        mask  = io_analyze.load(file_mask).astype(bool)
        #prior = np.load(file_prior)
        labelset = np.load(outdir + '/labelset.npy')
        intensity = np.load(outdir + '/intensity.npy')
        data = np.load(outdir + '/data.npz')['data']
        variance = np.load(outdir + '/variance.npz')['variance']
        imask = np.load(outdir + '/imask.npz')['imask']
        prior = {
            'labelset': labelset,
            'data': data,
            'variance': variance,
            'imask': imask,
            'intensity': intensity,
            }
    else:
        if pca:
            _prior, mask = load_or_compute_prior_and_mask(test, fold=fold)
            generator = rwsegment_pca_prior.PriorGenerator(labelset, mask=mask)
        else:
            generator = rwsegment_prior.PriorGenerator(labelset)
 
        ntrain = 0
        for train in config.vols:
            if train in fold: continue
            logger.debug('load training img: {}'.format(train))
            
            ## segmentation
            file_seg = config.dir_reg + test + train + 'regseg.hdr'
            seg = io_analyze.load(file_seg)
            
            ## image (for intensity prior)
            file_im = config.dir_reg + test + train + 'reggray.hdr'
            im = io_analyze.load(file_im)
            
            generator.add_training_data(seg,image=im)
            ntrain += 1

        if not pca:
            from scipy import ndimage
            mask    = generator.get_mask()
            struct  = np.ones((7,)*mask.ndim)
            mask    = ndimage.binary_dilation(
                    mask.astype(bool),
                    structure=struct,
                    ).astype(bool)
         
                 
        prior = generator.get_prior(mask)
        #import ipdb; ipdb.set_trace()

        nlabel = len(labelset)
        segprior = np.zeros(mask.shape)
        segprior.flat[prior['imask']] = labelset[np.argmax(prior['data'],axis=0)]
            
        entropymap = np.zeros(mask.shape)
        entropymap.flat[prior['imask']] = np.sum(
            np.log(prior['data'] + 1e-10)*prior['data'],
            axis=0)
        entropymap = entropymap / np.log(nlabel) * 2**15
            
        #np.savez(file_prior,**prior)
        #np.savez_compressed(file_prior,**prior)
        np.save(outdir + 'labelset', prior['labelset'])
        np.save(outdir + 'intensity.npy', prior['intensity'])
        np.savez_compressed(outdir + 'imask.npz', imask=prior['imask'])
        np.savez_compressed(outdir + 'data.npz', data=prior['data'].astype(np.float32))
        np.savez_compressed(outdir + 'variance.npz', variance=prior['variance'].astype(np.float32))
        
        io_analyze.save(file_mask, mask.astype(np.int32))
        io_analyze.save(file_segprior, segprior.astype(np.int32))
        #io_analyze.save(file_entropymap, entropymap.astype(np.int32))
        
    return prior, mask
    
    
def compute_dice_coef(seg1, seg2, labelset=None):
    if labelset is None:
        lbset = np.union(np.unique(seg1), np.unique(seg2))
    else:
        lbset = np.asarray(labelset, dtype=int)
    
    seg1.flat[~np.in1d(seg1, lbset)] = -1
    seg2.flat[~np.in1d(seg2, lbset)] = -1
    
    dicecoef = {}
    for label in lbset:
        l1 = (seg1==label)
        l2 = (seg2==label)
        d = 2*np.sum(l1&l2)/(1e-9 + np.sum(l1) + np.sum(l2))
        dicecoef[label] = d
    return dicecoef
    
