import sys
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
        self.nlabel = len(self.labelset)
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
        
        # compute orientations
        self.orients = np.asarray([
            [1,    0,    0   ],
            [0.7,  0,    0.7 ],
            [0.7,  0.7,  0   ],
            [0.7,  0 ,   -0.7],
            [0.7,  -0.7, 0   ],
            [0,    0,    1   ],
            [0,    1,    0   ],
            [0,    0,   -1  ],
            [0,    -1,   0   ],
            [-0.7, 0,    0.7 ],
            [-0.7, 0.7,  0   ],
            [-0.7, 0,    -0.7],
            [-0.7, -0.7, 0   ],
            [-1,   0,    0   ],
            ])
 
        self.orients /= np.c_[np.sqrt(np.sum(self.orients**2,axis=1))]
        self.orient_indices = np.arange(14*14).reshape((14,14))

        #nlabel = len(self.labelset)
        #indices_pairs = np.argwhere(np.triu(np.ones((nlabel,nlabel),dtype=int)))
        #self.label_pairs = self.labelset[indices_pairs]
        #self.nlabel_pair = len(self.label_pairs)
        #self.orient_indices = np.zeros((nlabel,nlabel), dtype=int)
        #self.orient_indices[tuple(indices_pairs.T)] = np.arange(indices_pairs.shape[0])
        #self.orient_indices += np.tril(self.orient_indices.T,k=-1)

    def compute_mask(self, train):
        seg = io_analyze.load(config.dir_reg + test + train + 'regseg.hdr')
        seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
        import ndimage 
        struct  = np.ones((20,)*mask.ndim)
        mask    = ndimage.binary_dilation(
                seg>0,
                structure=struct,
                )
        return mask

    def get_orient_scores(self,vecs):
        scores_ = np.dot(vecs, self.orients.T)
        scores  = 1./(1 + np.exp(-10*(scores_ - 0.8)))
        scores /= np.c_[np.sum(scores,axis=1)]
        return scores

    def load_or_compute_orientations(self,train,test, mask=None):
        from rwsegment import boundary_utils
        reload(boundary_utils)

        ## Train classifier
        logger.info('train orientations with train {} for test {}'.format(train, test))

        ##load image and seg
        seg = io_analyze.load(config.dir_reg + test + train + 'regseg.hdr').astype(int)
        seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
        
        ## sample points
        points = boundary_utils.sample_points(np.ones(seg.shape), 5,  mask=mask)
        logger.debug('number of sampled points = {}'.format(len(points)))


        ## find edges between muscles
        points_label = seg[tuple(points.T)]
        nlabel = len(self.labelset)
        hist = {}
        orient_scores = np.zeros((self.orients.shape[0],self.nlabel**2))
        #orient_prior = np.ones((3,self.label_pairs.shape[0]))
        ipair = 0
        for l1 in range(nlabel):
            label1 = self.labelset[l1]
            inds1 = np.where(points_label==label1)[0]
            hist[label1] = {}
            for l2 in range(nlabel):
                label2  = self.labelset[l2]
                inds2   = np.where(points_label==label2)[0]
                edges   = np.argwhere(np.triu(np.ones((inds1.size, inds2.size)),k=1))
                edges   = np.c_[inds1[edges[:,0]], inds2[edges[:,1]]]
                vecs    = points[edges[:,1]] - points[edges[:,0]]
                vecs    = vecs / np.c_[np.sqrt(np.sum(vecs**2,axis=1))]
                if l1==0 or l1==l2:
                    avg = np.ones(len(self.orients))/float(len(self.orients))
                else:
                    scores  = self.get_orient_scores(vecs)
                    avg     = np.mean(scores,axis=0)
                avgvecs = np.mean(vecs,axis=0)
                
                hist[label1][label2] = avg
                orient_scores[:,ipair] = avg
                #orient_prior[:,ipair] = avgvecs
                #print self.label_pairs[ipair], avg
                ipair += 1
                

        #orient_scores /= np.c_[np.sum(orient_scores,axis=1)]
        #import ipdb; ipdb.set_trace()
        return orient_scores,hist

    def load_or_compute_classifier(self,train,test,mask=None):
        from rwsegment import boundary_utils
        reload(boundary_utils)

        #idir_prior = config.dir_prior_edges + test
        #if not os.path.isdir(dir_prior):
        #    os.makedirs(dir_prior)
 
        ## Train classifier
        logger.info('train classifier with train {} for test {}'.format(train, test))

        ##load image and seg
        im = io_analyze.load(
            config.dir_reg + test + train + 'reggray.hdr').astype(float)
        nim = im/np.std(im)
        seg = io_analyze.load(config.dir_reg + test + train + 'regseg.hdr')
        seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
        
        ## sample points
        points = boundary_utils.sample_points(im, self.step,  mask=mask)
        logger.debug('number of sampled points = {}'.format(len(points)))

        #impoints = np.zeros(im.shape,dtype=int)
        #impoints[tuple(points.T)] = np.arange(len(points)) + 1

        ## compute edges
        edges,edgev,labels = boundary_utils.get_edges(im, points,  mask=mask)
        logger.debug('number of edges = {}'.format(len(edges)))

        ## extract profiles
        profiles,emap,dists = boundary_utils.get_profiles(nim, points, edges, rad=0)
        logger.debug('extracted profiles')

        ## make features
        x = boundary_utils.make_features(profiles, size=self.sizex, additional=[dists,edgev,edgev/dists])
        logger.debug('features made, size = {}'.format(len(x[0])))

        ## make annotations
        z = boundary_utils.is_boundary(points, edges, seg)
        logger.debug('annotations made')
        

        ## learn profiles
        logger.debug('training classifier')
        classifier = boundary_utils.Classifier()
        classifier.train(x,z)
        
        ## test classification
        logger.debug('testing classifier')
        cl, scores = classifier.classify(x)

        logger.info('non boundary correct rate: {:.3}'.format( 
            np.sum((np.r_[cl]==0)&(np.r_[z]==0))/np.sum(np.r_[z]==0).astype(float)))
        logger.info('boundary correct rate: {:.3}'.format( 
            np.sum((np.r_[cl]==1)&(np.r_[z]==1))/np.sum(np.r_[z]==1).astype(float)))
        

        ## store classifier
        #np.savetxt(dir_prior + 'classifier.txt', classifier.w)
        return classifier    


    def distance_to_train(self,train,points,mask=None):
        from scipy import ndimage
        labels = train[tuple(points.T)]
        ilabels = np.digitize(labels, self.labelset) - 1
        dist = np.zeros((len(points),self.nlabel))
        
        for i in range(self.nlabel):
            bin = train==self.labelset[i]
            bin = bin - ndimage.binary_erosion(bin)
            trainpts = np.argwhere(bin)
            for ip in range(len(points)):
                d = np.min(np.sqrt(np.sum((points[ip] - trainpts)**2,axis=1)))
                dist[ip,i] = d

        dist[np.arange(len(points)),ilabels] = 0
        return dist

    def process_sample(self,train, test):
        outdir = config.dir_work + 'autoseeds/' + config.basis + '/' + train + '/' + test
        logger.info('saving data in: {}'.format(outdir))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        ## get prior
        from scipy import ndimage
        segtrain = io_analyze.load(config.dir_reg + test + train + '/regseg.hdr')
        segtrain.flat[~np.in1d(segtrain, self.labelset)] = self.labelset[0]
        struct  = np.ones((10,)*segtrain.ndim)
        mask    = ndimage.binary_dilation(
                segtrain>0,
                structure=struct,
                ).astype(bool)
 
        #prior, mask = load_or_compute_prior_and_mask(
        #    test,force_recompute=self.force_recompute_prior)
        #mask = mask.astype(bool)

        ## load image
        file_name = config.dir_reg + test + 'gray.hdr'        
        logger.info('segmenting data: {}'.format(file_name))
        im      = io_analyze.load(file_name).astype(float)
        file_gt = config.dir_reg + test + 'seg.hdr'
        seg     = io_analyze.load(file_gt)
        seg.flat[~np.in1d(seg, self.labelset)] = self.labelset[0]
        
           
        ## normalize image
        nim = im/np.std(im)
     

        #orient_scores = self.load_or_compute_orientations(train,test, mask=mask)
 
        if 1:#not os.path.isfile(outdir + 'points.npy'):
  
            from rwsegment import boundary_utils
            reload(boundary_utils)
            ## sample points
            points = boundary_utils.sample_points(im, self.step,  mask=mask)
            points = points[mask[tuple(points.T)]]
            impoints = np.zeros(im.shape,dtype=int)
            impoints[tuple(points.T)] = np.arange(len(points)) + 1
            ipoints = np.where(impoints.ravel())[0]
            points = np.argwhere(impoints) 
            np.save(outdir + 'points.npy', points)
            impoints[tuple(points.T)] = np.arange(len(points)) + 1

            ## set unary potentials from prior: array of unary costs
            nlabel = len(self.labelset)
            dist = self.distance_to_train(segtrain, points)
            T = 10.0
            prob_pts = np.exp(-(dist/T)**2) / np.c_[np.sum(np.exp(-(dist/T)**2),axis=1)]
            #prob = np.c_[np.ones(im.size), np.zeros((im.size, nlabel-1))]
            #prob[mask.ravel(),:] = prior['data'].T
            #prob_pts = prob[ipoints,:]
            np.save(outdir + 'prob_points.npy', prob_pts) 
    
            ## binary potentials
            ## compute edges
            edges,edgev,labels = boundary_utils.get_edges(im, points,  mask=mask)
            edges = np.sort(edges,axis=1)
            np.save(outdir + 'edges.npy', edges)

            ## get orientation hist
            orient_scores,hist = self.load_or_compute_orientations(train,test, mask=mask)

            ##classify edges
            vecs = points[edges[:,1]] - points[edges[:,0]]
            vecs = vecs / np.c_[np.sqrt(np.sum(vecs**2,axis=1))] 
            scores = self.get_orient_scores(vecs)
            prob_orient = np.dot(scores, orient_scores)
            #prob_orient = prob_orient/np.c_[np.sum(prob_orient, axis=1)]
            np.save(outdir + 'prob_orient.npy', prob_orient) 

            ''' 
            ## load classifier
            classifier = self.load_or_compute_classifier(train,test, mask=mask)
 
            ## extract profiles
            profiles,emap,dists = boundary_utils.get_profiles(nim, points, edges, rad=0)
   
            ## make features  
            x = boundary_utils.make_features(
                profiles, 
                size=self.sizex, 
                additional=[dists,edgev,edgev/dists],
                )
            
            ## classify
            cl, scores = classifier.classify(x)

            ## ground truth
            z = boundary_utils.is_boundary(points, edges, seg)

            logger.info('non boundary classification: {}%'\
                .format(np.sum((np.r_[z]==0)*(np.r_[cl]==0))/float(np.sum(np.r_[z]==0))*100))
            logger.info('boundary classification: {}%'\
                .format(np.sum((np.r_[z]==1)*(np.r_[cl]==1))/float(np.sum(np.r_[z]==1))*100))
            np.save(outdir + 'classified.npy', cl) 

            ## probabilities
            prob_edges = 1.  - scores/np.c_[np.sum(scores, axis=1)]
      
            ##save probs
            np.save(outdir + 'prob_edges.npy',prob_edges)
            '''
        else:
            points     = np.load(outdir + 'points.npy')
            edges      = np.load(outdir + 'edges.npy')
            cl         = np.load(outdir + 'classified.npy') 
            prob_pts   = np.load(outdir + 'prob_points.npy')
            #prob_edges = np.load(outdir + 'prob_edges.npy')
            prob_orient = np.load(outdir + 'prob_orient.npy') 

        ## make potentials
        unary  = - np.log(prob_pts + 1e-10)
        #binary = - np.log(prob_edges + 1e-10)
        #thresh = (prob_orient.shape[1] - 1.0)/prob_orient.shape[1]
        thresh = (len(self.orients) - 1.0) / len(self.orients)
        orient_cost = - np.log(np.clip(prob_orient + thresh,0,1) + 1e-10)*100
        orient_cost = np.clip(orient_cost, 0, 1e10)
        #import ipdb; ipdb.set_trace()

        ## solve MRF:
        import ipdb; ipdb.set_trace()
        '''
        from rwsegment.mrf import fastPD
        class CostFunction(object):
            def __init__(self,**kwargs):
                self.binary = kwargs.pop('binary',0)
                self.orient_indices = kwargs.pop('orient_indices')
                self.orient_cost = kwargs.pop('orient_cost')

            def __call__(self,e,l1,l2):
                idpair = self.orient_indices[l1,l2]
                pair_cost = self.orient_cost[e,idpair]
                cost = (l1!=l2)*pair_cost
                #return (l1!=l2)*(1-cl[e])*0.1
                #return (l1!=l2)*self.binary[e,1]*0.1
                #y = l1!=l2
                #return self.binary[e, y]*pair_cost
                print e, l1, l2, cost
                return cost
 
        #sol, en = fastPD.fastPD_callback(unary, edges, cost_function(binary), debug=True)  
        cost_function = CostFunction(
            #binary=binary,
            orient_indices=self.orient_indices,
            orient_cost=orient_cost,
            )
        sol, en = fastPD.fastPD_callback(unary, edges, cost_function, debug=True)  
        '''
        wpairs = orient_cost
        from rwsegment.mrf import trw
        sol, en = trw.TRW_general(
            unary, edges, wpairs, niters=1000, verbose=True)

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
        for train in sample_list:
            for test in sample_list:
                if test==train: continue
                self.process_sample(train,test)
            

            
            
if __name__=='__main__':
    ''' start script '''
    sample_list = ['M44/', 'F32/']
    #sample_list = config.vols
   
    # Autoseeds
    if '-s' in sys.argv:
        segmenter = Autoseeds()
        segmenter.process_all_samples(sample_list)
    else:
        print 'doing nothing'
    

    
    
     
