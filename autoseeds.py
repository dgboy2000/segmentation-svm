import os
import numpy as np
from scipy import ndimage

from rwsegment import io_analyze
from rwsegment import rwsegment
from rwsegment.rwsegment import  BaseAnchorAPI
reload(rwsegment)

import segmentation_utils 
reload(segmentation_utils)
from segmentation_utils import load_or_compute_prior_and_mask
from segmentation_utils import compute_dice_coef
    
import config
reload(config)

from rwsegment import utils_logging
logger = utils_logging.get_logger('autoseeds',utils_logging.DEBUG)

class Autoseeds(object):
    
    def __init__(self):
        
        self.labelset  = np.asarray(config.labelset)
        self.force_recompute_prior = False
        self.force_recompute_edges = False
        
        self.step = [5,10,10]
        self.sizex = 24
 
        self.params  = {
            'beta'             : 50,     # contrast parameter
            'return_arguments' :['image'],
            
            # optimization parameter
            'per_label': True,
            'optim_solver':'unconstrained',
            'rtol'      : 1e-6,
            'maxiter'   : 2e3,
            }
    
    def load_or_compute_classifier(self,test):
        from rwsegment import boundary_utils
        reload(boundary_utils)

        dir_prior = config.dir_prior_edges + test
        if not os.path.isdir(dir_prior):
            os.makedirs(dir_prior)

        if not self.force_recompute_edges and \
                os.path.isfile(dir_prior + 'classifier.txt'):
            compute = False
            w = np.loadtxt(dir_prior + 'classifier.txt')
            classifier = boundary_utils.Classifier(w=w)
            return classifier
        else:
            ## Train classifier
            logger.info('train classifier for test {}'.format(test))
            Z = []
            X = []
            for train in config.vols:
                if train==test: continue

                ##load image and seg
                im = io_analyze.load(
                    config.dir_reg + test + train + 'reggray.hdr').astype(float)
                nim = im/np.std(im)
                seg = io_analyze.load(config.dir_reg + test + train + 'regseg.hdr')
                seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
                prior, mask = load_or_compute_prior_and_mask(
                    test, force_recompute=self.force_recompute_prior)
                
                ## sample points
                points = boundary_utils.sample_points(im, self.step,  mask=mask)
                logger.debug('number of sampled points = {}'.format(len(points)))

                ## compute edges
                edges,edgev,labels = boundary_utils.get_edges(im, points,  mask=mask)
                logger.debug('number of edges = {}'.format(len(edges)))

                ## extract profiles
                profiles,emap = boundary_utils.get_profiles(nim, points, edges, rad=0)
                logger.debug('extracted profiles')

                ## make features
                x = boundary_utils.make_features(profiles, size=self.sizex)
                logger.debug('features made, size = {}'.format(len(x[0])))

                ## make annotations
                z = boundary_utils.is_boundary(points, edges, seg)
                logger.debug('annotations made')
                
                X.extend(x)
                Z.extend(z)
                break

            ## learn profiles
            logger.debug('training classifier')
            classifier = boundary_utils.Classifier()
            classifier.train(X,Z)
            
            ## test classification
            logger.debug('testing classifier')
            cl, scores = classifier.classify(X)

            logger.info('non boundary : {:.3}'.format( 
                np.sum((np.r_[cl]==0)&(np.r_[Z]==0))/np.sum(np.r_[Z]==0).astype(float)))
            logger.info('boundary : {:.3}'.format( 
                np.sum((np.r_[cl]==1)&(np.r_[Z]==1))/np.sum(np.r_[Z]==1).astype(float)))
            
            ## store classifier
            np.savetxt(dir_prior + 'classifier.txt', classifier.w)
            return classifier    

    def process_sample(self,test):
        outdir = config.dir_work + 'autoseeds/' + config.basis + '/' + test
        logger.info('saving data in: {}'.format(outdir))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        ## get prior
        prior, mask = load_or_compute_prior_and_mask(
            test,force_recompute=self.force_recompute_prior)
        mask = mask.astype(bool)

        ## load image
        file_name = config.dir_reg + test + 'gray.hdr'        
        logger.info('segmenting data: {}'.format(file_name))
        im      = io_analyze.load(file_name).astype(float)
        file_gt = config.dir_reg + test + 'seg.hdr'
        seg     = io_analyze.load(file_gt)
        seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
        
           
        ## normalize image
        nim = im/np.std(im)
       
        if  not os.path.isfile(outdir + 'points.npy'):
  
            from rwsegment import boundary_utils
            reload(boundary_utils)
            ## sample points
            points = boundary_utils.sample_points(im, self.step,  mask=mask, maxiter=0)
            np.save(outdir + 'points.npy', points)

            impoints = np.zeros(im.shape,dtype=int)
            impoints[tuple(points.T)] = np.arange(len(points)) + 1
            ipoints = np.where(impoints.ravel())[0]

            ## set unary potentials from prior: array of unary costs
            nlabel = len(self.labelset)
            prob = np.c_[np.ones(im.size), np.zeros((im.size, nlabel-1))]
            prob[mask.ravel(),:] = prior['data'].T
 
            prob_pts = prob[ipoints,:]
            np.save(outdir + 'prob_points.npy', prob_pts) 
    
            ## binary potentials
            ## load classifier
            classifier = self.load_or_compute_classifier(test)
    
            ## compute edges
            edges,edgev,labels = boundary_utils.get_edges(im, points,  mask=mask)
            np.save(outdir + 'edges.npy', edges)

            ## extract profiles
            profiles,emap = boundary_utils.get_profiles(nim, points, edges, rad=0)

            ## make features  
            x = boundary_utils.make_features(profiles, size=self.sizex)
            
            ## classify
            cl, scores = classifier.classify(x)
            z = boundary_utils.is_boundary(points, edges, seg)
            logger.info('err in no boundary classification: {}%'\
                .format(np.sum((np.r_[z]==0)*(np.r_[cl]==1))/float(np.sum(np.r_[z]==0))*100))
            logger.info('err in boundary classification: {}%'\
                .format(np.sum((np.r_[z]==1)*(np.r_[cl]==0))/float(np.sum(np.r_[z]==1))*100))
            np.save(outdir + 'classified.npy', cl) 

            ## probabilities
            prob_edges = 1.  - scores/np.c_[np.sum(scores, axis=1)]
      
            ##save probs
            np.save(outdir + 'prob_edges.npy',prob_edges)
        else:
            points     = np.load(outdir + 'points.npy')
            edges      = np.load(outdir + 'edges.npy')
            cl         = np.load(outdir + 'classified.npy') 
            prob_pts   = np.load(outdir + 'prob_points.npy')
            prob_edges = np.load(outdir + 'prob_edges.npy')

        ## make potentials
        unary  = - np.log(prob_pts + 1e-10)
        binary = - np.log(prob_edges + 1e-10)

        ## solve MRF

        from rwsegment.mrf import fastPD

        class cost_function(object):
            def __init__(self, binary):
                self.binary = binary
            def __call__(self,e,l1,l2):
                #return (l1!=l2)*(1-cl[e])*1
                #return (l1!=l2)*self.binary[e,1]*0.1
                y = l1!=l2
                return self.binary[e, y]
 
        sol, en = fastPD.fastPD_callback(unary, edges, cost_function(binary), debug=True)  
        
        labels = self.labelset[sol]
        imsol = np.ones(im.shape, dtype=np.int32)*20
        imsol[tuple(points.T)] = labels
        io_analyze.save(outdir + 'imseeds.hdr', imsol)

        ## classify sol
        gtlabels    = seg[tuple(points.T)]
        priorlabels = self.labelset[np.argmin(unary,axis=1)]
        err_prior = 1 - np.sum(gtlabels==priorlabels)/float(len(points))
        err       = 1 - np.sum(gtlabels==labels)/float(len(points))
        logger.info('error in prior sol: {}%'.format(err_prior*100))
        logger.info('error in sol: {}%'.format(err*100))

        import ipdb; ipdb.set_trace()

        ## start segmenting
        sol,y = rwsegment.segment(
            nim, 
            seeds=seeds, 
            labelset=self.labelset, 
            weight_function=self.weight_function,
            **self.params
            )

       
        ## compute Dice coefficient per label
        dice    = compute_dice_coef(sol, seg,labelset=self.labelset)
        logger.info('Dice: {}'.format(dice))
        
        if not config.debug:
            io_analyze.save(outdir + 'sol.hdr', sol.astype(np.int32))
            np.savetxt(
                outdir + 'dice.txt', np.c_[dice.keys(),dice.values()],fmt='%d %.8f')
        
    def process_all_samples(self,sample_list):
        for test in sample_list:
            self.process_sample(test)
            

            
            
if __name__=='__main__':
    ''' start script '''
    sample_list = ['M44/']
    #sample_list = config.vols
   
    # Autoseeds
    segmenter = Autoseeds()
    segmenter.process_all_samples(sample_list)
    

    
    
     
