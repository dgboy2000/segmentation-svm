'''
    Work in progress code.
    Do not copy, do not distribute without permission !
    
    Pierre-Yves Baudin 2012
'''


import sys
import os
import numpy as np


from scipy import sparse
from scipy.sparse import linalg as splinalg


## compute prior 
class PriorGenerator:
    ''' generate prior
    '''
    
    def __init__(self, labelset):
        self.x0     = 0
        self.x02    = 0
        self.ntrain = 0
        self.mask   = 0
        
        self.labelset = np.asarray(labelset)
        
        
    def add_training_data(self, atlas):
        a = np.asarray(atlas, dtype=int)
        
        ## set unwanted labels to background label (labelset[0])
        bg = self.labelset[0]
        a.flat[~np.in1d(a, self.labelset)] = bg
        
        ## compute background mask
        self.mask = self.mask | (a!=bg)
        
        ## compute average
        x = (np.c_[a.ravel()]==self.labelset).astype(float)
        self.x0     = self.x0 + x
        self.x02    = self.x02 + x**2
        self.ntrain += 1
        
    def get_mask(self):
        return self.mask
        
    def get_prior(self, mask=None):
        assert np.all([self.mask.shape[d]==mask.shape[d] \
            for d in range(mask.ndim)])
                
        if mask is not None:
            fmask = mask.ravel()
        else:
            fmask = np.ones(self.mask.shape,dtype=bool).ravel()
        
        nlabel = len(self.labelset)
        ij     = np.where(np.c_[fmask]*np.ones((1,nlabel)))
        mean   = self.x0[fmask,:].ravel() / np.float(self.ntrain)
        var    = (self.x02 - self.x0**2)
        var    = var[fmask,:].ravel() / np.float(self.ntrain)
        
        return {
            'ij'    : ij,
            'mean'  : mean,
            'var'   : var,
            }
        
##------------------------------------------------------------------------------
def segment_mean_prior(
        image,
        prior,
        seeds=[],
        cmap=None,
        add_linear_term=None,
        weight_function=None,
        laplacian=None,
        return_laplacian=False,
        return_arguments=['solution'],
        **kwargs
        ):
    ''' segment an image with method from MICCAI'12
    
    arguments:
        image   = image to segment
        prior   = prior probability values as a dict: {'ij':ij, 'mean':data}
                
    keyword arguments:
        seeds    = label map, with -1 for unmarked pixels
                 must have the same shape as image
        labelset = output labels (in the order of prior)
                 seed labels not in labelset are ignored
        cmap     = confidence map, with shape = image.shape,
                 such that:
                    cmap[i]~0 if we should rely more on the image
                    cmap[i]~1 if we should rely more on the prior model
        add_linear_term = additional linear term (for modifying the functional)
        lmbda   = weight for prior
        beta    = contrast parameter if no weight_function is provided
        weight_function = user provided function:
            (ij, data) = weight_function(image)
        laplacian = (L,border,B) replace internal laplacian computing
        add 'laplacian' to return_arguments
        
                
    '''
    
    ## constants
    # assume prior has shape (nvar, nlabel)
    nlabel      = np.max(prior['ij'][1]) + 1 
    labelset    = np.array(kwargs.pop('labelset', range(nlabel)), dtype=int)
    inds        = np.arange(image.size)
    marked      = np.where(np.in1d(seeds,labelset))[0]
    unknown     = np.setdiff1d(inds,marked, assume_unique=True)
    nvar        = unknown.size
    
    ## unknown pixels with no prior as set to 1./nlabels
    noprior = np.setdiff1d(prior['ij'][0], inds, assume_unique=True)
    noprior = np.intersect1d(noprior, unknown, assume_unique=True)
    nnoprior = len(noprior)
    pr_ij   = np.append(
        prior['ij'], 
        [np.tile(noprior,nlabel),[x%nnoprior for x in range(nnoprior*nlabel)]], 
        axis=1,
        ).astype(int)
    pr_data = np.append(prior['mean'], [1.0/nlabel]*nnoprior*nlabel)
        
    ## parameters
    beta    = kwargs.pop('beta', 1.)
    lmbda   = kwargs.pop('lmbda', 1.) * np.ones(nlabel)
    per_label = kwargs.pop('per_label',True)
    
    ## can be scipy or mosek
    optim_solver = kwargs.pop('optim_solver','scipy')

    
    ## compute laplacian
    if laplacian is None:
        logger.info('compute laplacian')
            
        L, border, B = compute_laplacian(
            image,
            marked=marked,
            beta=beta, 
            weight_function=weight_function,
            )
    else:
        L,border,B = laplacian

    ## prior matrix
    pmat = sparse.coo_matrix(
        (pr_data,pr_ij),
        shape=(image.size,nlabel),
        ).tocsr()
        
    if not per_label:
        ## compute all in one pass
        
        ## intermediary matrices
        LL  = sparse.kron(np.eye(nlabel), L)
        BB  = sparse.kron(np.eye(nlabel), B)
        D   = sparse.kron(sparse.eye(nvar,nvar), np.diag(lmbda))
        if cmap is not None:
            D = D * sparse.kron(
                sparse.eye(nlabel, nlabel),
                sparse.spdiags(cmap.flat[unknown], 0, nvar, nvar),
                )

        x0 = sparse.bmat([[pmat[unknown, il]] for il in range(nlabel)])
        xm = (np.c_[seeds.flat[border]]==labelset).T\
            .reshape((BB.shape[1],1))
            
        ## additional linear term
        Y = 0
        if add_linear_term is not None:
            linterm = add_linear_term.reshape((-1,nlabel), order='F')
            Y = linterm[unknown,:].reshape((-1,1),order='F')
            
        ## solve
        logger.info('solve rw')
        P = LL + D
        q = BB*xm - D*x0 + Y
        if P.nnz==0:
            logger.warning('in QP, P=0. Returning 1-(q>0)') 
            ## assumes binary q. If not binary, set to q>epsilon ?
            x = (1 - (q>0))/(nlabel - 1)
        else:
            if optim_solver=='mosek':
                x = solve_qp_mosek(P,q,nlabel)
            else:
                x = solve_qp(P, q, **kwargs)
        
    else:
        ## compute separately for each label (best)
        
        if cmap is not None:
            Cmap = sparse.spdiags(cmap.flat[unknown], 0, nvar, nvar)
        x = np.zeros((nvar, nlabel))
        
        if add_linear_term is not None:
            linterm = add_linear_term.reshape((-1,nlabel), order='F')
            linterm = np.asmatrix(linterm[unknown,:])
        
        for il in range(nlabel):
            D = lmbda[il]*sparse.eye(nvar,nvar)
            if cmap is not None:
                D = D * Cmap
            x0 = pmat[unknown, il]
            xm = np.c_[seeds.flat[border]==il]
            
            ## additional linear term
            Y = 0
            if add_linear_term is not None:
                Y = linterm[:,il]
            
            P = L + D
            q = B*xm - D*x0 + Y
            # import ipdb; ipdb.set_trace()
            if P.nnz==0:
                logger.warning(
                    'label #{}. In QP, P=0. Returning (1-q>0)/nlabel'.format(il)
                    ) 
                ## assumes binary q. If not binary, set to q>epsilon ?
                _x = (1 - (q>0))/(nlabel - 1)
            else:
                _x = solve_qp(P, q, **kwargs)
            x[:,il] = _x.ravel()

        
    ## reshape solution
    # sol = np.zeros((image.size,nlabel))
    sol = (np.c_[seeds.ravel()]==labelset).astype(float)
    sol[unknown,:] = x
    sol = sol.ravel('F')
    # import ipdb; ipdb.set_trace()

    ## output arguments
    rargs = []
    for arg in return_arguments:
        if   arg=='solution':  rargs.append(sol)
        elif arg=='laplacian': rargs.append((L,border,B))
        else: pass
    
    if len(rargs)==1: return rargs[0]
    else: return tuple(rargs)
    
##------------------------------------------------------------------------------
def solve_qp(P,q, maxiter=1e2, rtol=1e-3):
    '''
        solve: min(X): Xt*P*X + 2Xt*q + cst
    '''
    tol = np.max(P.data)*rtol
    
    if P.nnz==0:
        logger.error('P has no non-zeros entries')
        return np.zeros(q.size)
    
    maxiter = int(maxiter)
    if 0:#'use_cvxopt':
        import cvxopt
        from cvxopt import solvers
        
        solvers.options['show_progress'] = True
        solvers.options['maxiters'] = maxiter
        solvers.options['abstol'] = tol
        solvers.options['feastol'] = tol
        solvers.options['reltol'] = tol
        
        row,col = P.nonzero()
        _P = cvxopt.spmatrix(
            P.data.tolist(), 
            row.tolist(),
            col.tolist(),
            )
        _q = cvxopt.matrix(np.asarray(q).ravel())
        sol = solvers.qp(_P,_q)
        x = np.asarray(sol['x']).ravel()
        
    else:
        ## use standard conjugate gradient
        x,info = splinalg.cg(
            P,-q,
            maxiter=maxiter,
            tol=tol,
            )
        if info!=0:
            logger.error('QP did not converge. info={}'.format(info))
    return x

##------------------------------------------------------------------------------
def test_definiteness(P,numtest=100):
    n = P.shape[1]
    for t in range(numtest):
        vec = np.asmatrix(np.random.random((n,1))-0.5)
        val = vec.T*P*vec
        if val<0:
            import ipdb;ipdb.set_trace()
    
    
##------------------------------------------------------------------------------
def solve_qp_mosek(P,q,nlabel):
    import mosek
    import sys
    

    
    def streamprinter(text): 
        sys.stdout.write(text) 
        sys.stdout.flush()
    
    inf = 0
    
    # Open MOSEK and create an environment and task 
    # Make a MOSEK environment 
    env = mosek.Env () 
    
    # Attach a printer to the environment 
    env.set_Stream (mosek.streamtype.log, streamprinter) 
    
    # Create a task 
    task = env.Task() 
    
    # set convexity check to:
    # task.putintparam(mosek.iparam.check_convexity , 0) #none
    task.putintparam(mosek.iparam.check_convexity , 1) # simple
    
    task.set_Stream (mosek.streamtype.log, streamprinter)
    
    NUMVAR = q.size
    NUMCON = NUMVAR/nlabel
    NUMANZ = NUMVAR
    
    task.putmaxnumvar(NUMVAR) 
    task.putmaxnumcon(NUMCON) 
    task.putmaxnumanz(NUMANZ)
    
    task.append(mosek.accmode.con,NUMCON)
    task.append(mosek.accmode.var,NUMVAR)
    task.putcfix(0.0)
    
    # Set up and input bounds and linear coefficients
    logger.debug('set bounds on variables')
    bkx = [mosek.boundkey.ra for i in range(NUMVAR)]
    blx = [ 0.0 for i in range(NUMVAR)]
    bux = [ 1.0 for i in range(NUMVAR)]
    
    c = np.asarray(q).ravel()
    
    # test_definiteness(P,numtest=100)
    logger.debug('setup objective matrix')
    (qsubi,qsubj) = sparse.tril(P).nonzero()
    qval = sparse.tril(P).data
    
    
    ## set the constraints
    logger.debug('setup the probability constraints')
    
    A = sparse.bmat([[
        sparse.eye(NUMCON,NUMCON) for i in range(nlabel)
        ]])
    (asubi,asubj) = A.nonzero()
    aval = A.data
    
    # Set the bounds on variable j 
    logger.debug('add constraints into Mosek')
    # blx[j] <= x_j <= bux[j] 
    task.putboundlist(mosek.accmode.var,np.arange(NUMVAR),bkx,blx,bux)
    
    # Set the linear term c_j in the objective. 
    task.putclist(np.arange(NUMVAR),c)
    
    # Set up and input quadratic objective
    task.putqobj(qsubi,qsubj,qval)
    
    # Set up and input contraint matrix
    task.putaijlist(asubi,asubj,aval)
    
    # Input the objective sense (minimize/maximize) 
    task.putobjsense(mosek.objsense.minimize)
    

    # import ipdb; ipdb.set_trace()
    # Optimize 
    logger.info('start optimizing')
    task.optimize() 
    logger.info('done optimizing')
    
    # Print a summary containing information 
    # about the solution for debugging purposes 
    #task.solutionsummary(mosek.streamtype.msg)
    
    prosta = [] 
    solsta = [] 
    [prosta,solsta] = task.getsolutionstatus(mosek.soltype.itr)
    
    # Output a solution 
    xx = [0 for i in range(NUMVAR)] 
    task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0,NUMVAR, xx)
    
    return np.array(xx)

##------------------------------------------------------------------------------
def energy_mean_prior(
        prior,
        prob_segmentation,
        cmap=None,
        seeds=[],
        weight_function=None,
        laplacian=None,
        **kwargs
        ):
    
    ## constants
    # assume prior has shape (nvar, nlabel)
    nlabel      = np.max(prior['ij'][1]) + 1 
    labelset    = np.array(kwargs.pop('labelset', range(nlabel)), dtype=int)
    inds        = np.arange(prob_segmentation.size/nlabel)
    marked      = np.where(np.in1d(seeds,labelset))[0]
    unknown     = np.setdiff1d(inds,marked, assume_unique=True)
    nvar        = unknown.size
    
    ## unknown pixels with no prior as set to 1./nlabels
    noprior = np.setdiff1d(prior['ij'][0], inds, assume_unique=True)
    noprior = np.intersect1d(noprior, unknown, assume_unique=True)
    nnoprior = len(noprior)
    pr_ij   = np.append(
        prior['ij'], 
        [np.tile(noprior,nlabel),[x%nnoprior for x in range(nnoprior*nlabel)]], 
        axis=1,
        ).astype(int)
    pr_data = np.append(prior['mean'], [1.0/nlabel]*nnoprior*nlabel)
        

    ## prior matrix
    pmat = sparse.coo_matrix(
        (pr_data,pr_ij),
        shape=(inds.size,nlabel),
        ).tocsr()
    
    ## generate segmentation vector
    # segvar = segmentation.flat[unknown]
    # segvar[~np.in1d(segvar,labelset)] = labelset[0]
    # x = np.asarray(np.c_[segvar]==labelset, dtype=float)
    x = prob_segmentation.reshape((-1,nlabel), order='F')
    x = x[unknown,:]
    
    Cmap = 1
    if cmap is not None:
        Cmap = cmap.flat[unknown]
    
    en = 0
    for il in range(nlabel):
        x0 = np.asarray(pmat[unknown, il].todense()).ravel()
        en += np.sum(Cmap*(x[:,il]-x0)**2)
        
    return float(en)
    
##------------------------------------------------------------------------------
def energy_RW(
        image,
        lbelset,
        prob_segmentation,
        seeds=[],
        weight_function=None,
        laplacian=None,
        **kwargs
        ):
    ## constants
    lbset       = np.asarray(lbelset)
    nlabel      = len(lbset)
    inds        = np.arange(image.size)
    marked      = np.where(np.in1d(seeds,lbset))[0]
    unknown     = np.setdiff1d(inds,marked, assume_unique=True)
    nvar        = unknown.size
    beta    = kwargs.pop('beta', 1.)
    
    ## generate segmentation vector
    # segvar = segmentation.flat[unknown]
    # segvar[~np.in1d(segvar,labelset)] = labelset[0]
    # x = np.asmatrix(np.c_[segvar]==labelset, dtype=float)
    x = prob_segmentation.reshape((-1,nlabel), order='F')
    x = np.asmatrix(x[unknown,:])
    
    ## compute laplacian
    if laplacian is None:
        logger.info('compute laplacian')
        L, border, B = compute_laplacian(
            image,
            marked=marked,
            beta=beta, 
            weight_function=weight_function,
            )
    else:
        L,border,B = laplacian
        
    ## compute energy
    en = 0.0
    for il in range(nlabel):
        en += x[:,il].T * L * x[:,il]
    return float(en)
    
##------------------------------------------------------------------------------
def compute_laplacian(
        image, 
        marked=None,
        beta=1., 
        weight_function=None,
        verb=0,
        ):
    ''' compute laplacian matrix for using with Random Walks
    args:
        image
        marked  : indices of marked pixels (in image.flat)
        beta    : contrast parameter for default weight_function:
            for touching pixel pair (i,j),
            wij = exp (- beta (image.flat[i] - image.flat[j])^2)
            
        weight_function : user function to compute edge weights
            must return: (ij, data), 
            where ij[0][k]  = indices of vertex #0 in pair #k
                  ij[1][k]  = indices of vertex #1 in pair #k
                  data[k]   = weight of pair #k
            Pairs are included once: if (m,n) is included, (n,m) is not.
            
    return args:
        L       : laplacian matrix (for unmarked pixels only)
        
        Only if marked is provided:
        border  : indices of marked pixels touching unknown ones
        B       : laplacian matrix for (unknown, border) pairs
    
    '''
    
    im = np.asarray(image)
    N  = im.size
    
    ## compute weights
    logger.info('weight_function')
    if weight_function is None:
        # if no wf provided, use standard with provided beta
        weight_function = lambda x: weight_std(x,beta=beta)
    ij, data = weight_function(im)
    
    ## affinity matrix
    logger.info('laplacian matrix')
    A = sparse.coo_matrix((data, ij), shape=(N,N))
    A = A + A.T
    
    ## laplacian matrix
    L = (sparse.spdiags(A.sum(axis=0),0,N,N) - A).tocsr()
    
    ## return laplacian
    if marked is None:
        return L
        
    ## if marked is not None,
    ## keep only unknown pixels, and marked pixels touching unknown
    logger.info('decompose into Lu, B')
    unknown = np.setdiff1d(np.arange(im.size), marked, assume_unique=True)
    
    mask = np.ones(im.shape, dtype=int)
    mask.flat[unknown] = 0
    ijborder = (mask.flat[ij[0]] + mask.flat[ij[1]])==1
    mask.flat[ij[0][ijborder]] *= 2
    mask.flat[ij[1][ijborder]] *= 2
    border = np.where(mask.ravel()>=2)[0]

    Lu = L[unknown,:][:,unknown]
    B  = L[unknown,:][:,border]
    
    # import ipdb; ipdb.set_trace()
    
    return Lu,border,B
    
    
##------------------------------------------------------------------------------
def weight_std(image, beta=1.0):
    ''' standard weight function 
    
        for touching pixel pair (i,j),
            wij = exp (- beta (image.flat[i] - image.flat[j])^2)
    '''
    im = np.asarray(image)
    
    ## affinity matrix sparse data
    data = np.exp( - beta * np.r_[tuple([
        np.square(
            im.take(np.arange(im.shape[d]-1), axis=d).ravel() - \
            im.take(np.arange(1,im.shape[d]), axis=d).ravel(),
            )
        for d in range(im.ndim)
        ])])
    
    ## affinity matrix sparse indices
    inds = np.arange(im.size).reshape(im.shape)
    ij = (
        np.r_[tuple([inds.take(np.arange(im.shape[d]-1), axis=d).ravel() \
            for d in range(im.ndim)])],
        np.r_[tuple([inds.take(np.arange(1,im.shape[d]), axis=d).ravel() \
            for d in range(im.ndim)])],
        )
    
    return ij, data
    
##------------------------------------------------------------------------------
    
import logging
logger = logging.getLogger('rw logger')
loglevel = logging.WARNING
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
    ''' test segmentation '''
    import ioanalyze
    from scipy import ndimage
    
    ## build prior
    labelset  = [0,13,14,15,16]
    generator = PriorGenerator(labelset)
    slices = [slice(20,40),slice(None),slice(None)]
    
    ## load training data (0 is test data)
    for i in range(1,11):
    # for i in range(1,31):
        file_name = 'data/seg_{}.hdr'.format(i)
        print '  training data: {}'.format(file_name)
        seg = ioanalyze.load(file_name)[slices]
        generator.add_training_data(seg)

        
    ## get mask from prior data
    mask    = generator.get_mask()
    
    ## dilate mask
    struct  = np.ones((7,)*mask.ndim)
    mask    = ndimage.binary_dilation(
            mask.astype(bool),
            structure=struct,
            )
    ioanalyze.save('mask.hdr', mask.astype(np.int32))
    
    ## get prior
    prior = generator.get_prior(mask=mask)
    x0 = np.zeros((mask.size,len(labelset)))
    x0[:,0] = 1
    x0[prior['ij']] = prior['mean']
    mean = np.argmax(x0, axis=1).reshape(mask.shape)
    ioanalyze.save('mean.hdr', mean.astype(np.int32))
    
    ## segment image
    file_name = 'data/im_0.hdr'
    print 'segmenting data: {}'.format(file_name)
    
    im      = ioanalyze.load(file_name)[slices]
    seeds   = (-1)*mask
    params  = {
        'lmbda'     : 1e-2, # weight of prior
        'beta'      : 1e-2, # contrast parameter
        'tol'       : 1e-3, # optimization parameter
        'maxiter'   : 1e3,
        'verb'      : 1,    # verbosity
        }
    sol = segment_mean_prior(
        im, 
        prior, 
        seeds=seeds, 
        labelset=labelset, 
        **params
        )
    ioanalyze.save('solution.hdr', sol.astype(np.int32))
    
    