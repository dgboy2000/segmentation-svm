import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment
from rwsegment import weight_functions as wflib
from rwsegment import rwsegment_prior_models as prior_models
from rwsegment import loss_functions
from rwsegment import rwsegment_prior_models as models
reload(models)
import svm_rw_functions
from svm_rw_functions import MetaAnchorApi
reload(rwsegment)
reload(loss_functions)
reload(prior_models)


import segmentation_utils 
reload(segmentation_utils)

from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
    
import config
reload(config)

from rwsegment import utils_logging
logger = utils_logging.get_logger('segmentation_batch',utils_logging.INFO)

class SegmentationBatch(object):
    
    def __init__(
            self,
            laplacian_weights=1, 
            anchor_weights=None, 
            **kwargs):

        self.labelset  = np.asarray(config.labelset)
        self.model_name = kwargs.pop('name', None)
        self.force_recompute_prior = False
        self.sweights = kwargs.pop('sweights', 1)
        
        self.params  = {
            'beta'             : 100,     # contrast parameter
            'return_arguments' :['image','y'],
            # optimization parameter
            'per_label': True,
            'optim_solver':'unconstrained',
            'rtol'      : 1e-6,
            'maxiter'   : 2e3,
            }
        
        
        self.laplacian_functions = [
           {'func': lambda im,i,j: wflib.weight_std(im,i,j, beta=10)},
           {'func': lambda im,i,j: wflib.weight_std(im,i,j, beta=50)},
           {'func': lambda im,i,j: wflib.weight_std(im,i,j, beta=100)},
           ]

        self.prior_models = [
           {'name': 'constant',  'default': 0},
           {'name': 'entropy',   'default': 0},
           {'name': 'intensity', 'default': 0.0},
           {'name': 'variance',  'default': 0.0},
           {'name': 'constant cmap', 'default': 0.0},
           ]
        self.laplacian_weights = laplacian_weights
        self.anchor_weights = anchor_weights

        logger.info('Model name = {}, using prior weights={}'\
            .format(self.model_name, self.anchor_weights))
    
    def process_sample(self,test,fold=None):

        ## get prior
        prior, mask = load_or_compute_prior_and_mask(
            test,
            fold=fold,
            force_recompute=self.force_recompute_prior)
        seeds   = (-1)*mask
        
        ## load image
        file_name = config.dir_reg + test + 'gray.hdr'        
        logger.info('segmenting data: {}'.format(file_name))
        im      = io_analyze.load(file_name)
        file_gt = config.dir_reg + test + 'seg.hdr'
        seg     = io_analyze.load(file_gt)
        seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
        
           
        ## normalize image
        nim = im/np.std(im)

        ## laplacian functions
        nlabel = len(self.labelset)

        ## instantiate models
        imask = prior['imask']
        average = prior['data']
        variance = prior['variance']
        im_avg, im_var = prior['intensity']
        
        self.prior_models[0]['api'] = models.Constant(imask, average)
        self.prior_models[1]['api'] = models.Entropy_no_D(imask, average)
        #self.prior_models[1]['api'] = models.Spatial(imask, average, sweights=self.sweights)
        self.prior_models[2]['api'] = models.Intensity(im_avg, im_var)
        self.prior_models[3]['api'] = models.Variance_no_D(imask, average, variance=variance)
        self.prior_models[4]['api'] = models.Constant_Cmap(imask, average)
 
        ## init anchor_api
        laplacian_function = svm_rw_functions.LaplacianWeights(
            nlabel,self.laplacian_functions, weights=self.laplacian_weights) 
        anchor_api = svm_rw_functions.MetaAnchorApi(
            self.prior_models, weights=self.anchor_weights)

        ## start segmenting
        #import ipdb; ipdb.set_trace()
        sol,y = rwsegment.segment(
            nim, 
            anchor_api,
            seeds=seeds, 
            labelset=self.labelset, 
            laplacian_function=laplacian_function,
            **self.params
            )

        ## compute losses
        z = seg.ravel()==np.c_[self.labelset]
        flatmask = mask.ravel()*np.ones((len(self.labelset),1))
        
        ## loss 0 : 1 - Dice(y,z)
        loss0 = loss_functions.ideal_loss(z,y,mask=flatmask)
        logger.info('Tloss = {}'.format(loss0))
       
        ## compute Dice coefficient per label
        dice    = compute_dice_coef(sol, seg,labelset=self.labelset)
        logger.info('Dice: {}'.format(dice))
        
        if not config.debug:
            if fold is not None:
                test_name = 'f{}_{}'.format(fold[0][:2], test)
            else: test_name = test
            outdir = config.dir_seg + \
                '/{}/{}'.format(self.model_name,test_name)
            logger.info('saving data in: {}'.format(outdir))
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
        
            f = open(outdir + 'losses.txt', 'w')
            f.write('ideal_loss\t{}\n'.format(loss0))
            f.close()
            
            io_analyze.save(outdir + 'sol.hdr', sol.astype(np.int32)) 
            np.savetxt(
                outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %.8f')
        
    def compute_mean_segmentation(self, list, fold=None):
        for test in list:
            file_gt = config.dir_reg + test + 'seg.hdr'
            seg     = io_analyze.load(file_gt)
            seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
           
            ## get prior
            prior, mask = load_or_compute_prior_and_mask(
                test,force_recompute=self.force_recompute_prior, fold=fold)
            mask = mask.astype(bool)            

            y = np.zeros((len(self.labelset),seg.size))
            y[:,0] = 1
            y.flat[prior['imask']] = prior['data']
 
            sol = np.zeros(seg.shape,dtype=np.int32)
            sol[mask] = self.labelset[np.argmax(prior['data'],axis=0)]

            ## compute losses
            z = seg.ravel()==np.c_[self.labelset]
            flatmask = mask.ravel()*np.ones((len(self.labelset),1))
 
            ## loss 0 : 1 - Dice(y,z)
            loss0 = loss_functions.ideal_loss(z,y,mask=flatmask)
            logger.info('Tloss = {}'.format(loss0))
            
           ## compute Dice coefficient per label
            dice    = compute_dice_coef(sol, seg,labelset=self.labelset)
            logger.info('Dice: {}'.format(dice))
            
            if not config.debug:
                if fold is not None:
                    test_name = 'f{}_{}'.format(fold[0][:2], test)
                else: test_name = test
                outdir = config.dir_seg + \
                    '/{}/{}'.format('mean',test_name)
                logger.info('saving data in: {}'.format(outdir))
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                io_analyze.save(outdir + 'sol.hdr', sol.astype(np.int32)) 
                np.savetxt(
                    outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %.8f')
 

    def process_all_samples(self,sample_list, fold=None):
        for test in sample_list:
            self.process_sample(test, fold=fold)
            
            
if __name__=='__main__':
    ''' start script '''
    import sys
    if not '-s' in sys.argv:
        sys.exit()

    #w = [
    #    4.560767447457286525e-11,
    #    4.069170651869788340e-10,
    #    5.243733854153335372e+00,
    #    5.497790917889032036e-10,
    #    5.360258783602039125e-12,
    #    8.735299325670829729e-02,
    #    3.817273881679687959e-03, 
    #    ]

    #segmenter = SegmentationBatch(
    #    laplacian_weights=w[:4],
    #    anchor_weights=w[4:], 
    #    name='svm_C100_Cp1e-2')

<<<<<<< HEAD
    w = [0,1,0,1e-2,0,0,0,0]
    segmenter = SegmentationBatch(
        laplacian_weights=w[:3],
        anchor_weights=w[3:], 
        name='registration')
    for fold in config.folds:
        segmenter.compute_mean_segmentation(fold, fold=fold)

    #w = [0,1,0,1e-2,0,0,0,0]
    #segmenter = SegmentationBatch(
    #    laplacian_weights=w[:3],
    #    anchor_weights=w[3:], 
    #    name='constant1e-2')
    #segmenter.process_all_samples(config.vols.keys())
    #
    #w = [0,1,0,1e-2,0,1e-2,0,0]
    #segmenter = SegmentationBatch(
    #    laplacian_weights=w[:3],
    #    anchor_weights=w[3:], 
    #    name='constant1e-2Intensity1e-2')
    #segmenter.process_all_samples(config.vols.keys())
=======
    #w = [0,0,1,0,1e-2,0,0]
    #segmenter = SegmentationBatch(
    #    laplacian_weights=w[:4],
    #    anchor_weights=w[4:], 
    #    name='constant1e-2')
    #for fold in config.folds:
    #    segmenter.process_all_samples(fold, fold=fold)

    w = [0,0,1,0,0,1,0]
    segmenter = SegmentationBatch(
        laplacian_weights=w[:4],
        anchor_weights=w[4:], 
        name='entropy1e0')
    for fold in config.folds[1:2]:
        segmenter.process_all_samples(fold, fold=fold)
>>>>>>> 11a33b11ae9e2d6c594e9402cb223b96ce2500db

    #w = [0,1,0,0,1e-2,0,0,0]
    #segmenter = SegmentationBatch(
    #    laplacian_weights=w[:3],
    #    anchor_weights=w[3:], 
    #    name='entropy1e-2')
    #segmenter.process_all_samples(config.vols.keys())
    ##
    #w = [0,1,0,0,0,0,1e-1,0]
    #segmenter = SegmentationBatch(
    #    laplacian_weights=w[:3],
    #    anchor_weights=w[3:], 
    #    name='variance1e-1')
    #segmenter.process_all_samples(config.vols.keys())

    #w = [0,1,0,0,0,0,0,1e-2]
    #segmenter = SegmentationBatch(
    #    laplacian_weights=w[:3],
    #    anchor_weights=w[3:],                            
    #    name='constantcmap1e-2')
    #segmenter.process_all_samples(config.vols.keys())




