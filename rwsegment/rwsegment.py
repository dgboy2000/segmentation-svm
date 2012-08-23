'''
    Work in progress code.
    Do not copy, do not distribute without permission !
    
    Pierre-Yves Baudin 2012
'''


import sys
import os
import numpy as np


from scipy import sparse
        
import utils_logging
# logger = utils_logging.get_logger('rwsegment',utils_logging.INFO)
logger = utils_logging.get_logger('rwsegment',utils_logging.DEBUG)
        
class BaseAnchorAPI(object):
    def __init__(self,anchor, anchor_weight=1.0, **kwargs):
        self.anchor = anchor
        self.labelset = anchor['labelset']
        self.imask = anchor['imask']
        self.anchor_weight = anchor_weight
        
    def get_labelset(self):
        return self.labelset
    
    def get_anchor_and_weights(self,D):
        nlabel = len(self.labelset)
        weights = self.anchor_weight * np.ones((nlabel,D.size)) * D
        return self.anchor, weights
        
##------------------------------------------------------------------------------
def segment(
        image,
        anchor_api,
        seeds=[],
        weight_function=None,
        return_arguments=['image'],
        **kwargs
        ):
    ''' segment an image with method from MICCAI'12
    
    arguments:
        image   = image to segment
        anchor_api:
            labelse = api.get_labelset()
            anchor, weights = api.get_anchor_and_weights(D)
            anchor is a dict
        
    keyword arguments:
        
    
        seeds    = label map, with -1 for unmarked pixels
                 must have the same shape as image
                 
        anchor_weights = a list of vectors of size npixel
        
        beta    = contrast parameter if no weight_function is provided
        weight_function = user provided function:
            (ij, data) = weight_function(image)
        add 'laplacian' to return_arguments
        
                
    '''
    
    ## constants
    labelset    = anchor_api.get_labelset()
    nlabel      = len(labelset) 
    npixel      = image.size
    inds        = np.arange(npixel)
    marked      = np.where(np.in1d(seeds,labelset))[0]
    unknown     = np.setdiff1d(inds,marked, assume_unique=True)
    nunknown    = unknown.size
        
    ## parameters
    beta    = kwargs.pop('beta', 1.)
    wanchor   = kwargs.pop('wanchor', 1.) # TODO: consolidate this as part of anchor function
    per_label = kwargs.pop('per_label',True)
    
    ## ground truth
    ground_truth = kwargs.pop('ground_truth',None)
    if ground_truth is not None:
        ground_truth[~np.in1d(ground_truth,labelset)] = labelset[0]

    ## compute laplacian
    logger.debug('compute laplacian')
        
    Lu, border, B,D = compute_laplacian(
        image,
        marked=marked,
        beta=beta, 
        weight_function=weight_function,
        return_D=True,
        )

    ## anchor function:
    anchor, anchor_weights = anchor_api.get_anchor_and_weights(D)
    # anchor, anchor_weights = anchor_api.get_anchor_and_weights(1)
        
    ## per label lists of vectors
    list_x0, list_Omega, list_xm, list_GT = [],[],[],[]
    for l in range(nlabel):
        x0 = 1/float(nlabel) * np.ones(npixel)
        x0[anchor['imask']] = anchor['data'][l]
        list_x0.append(x0[unknown])
        if seeds!=[]:
            list_xm.append(seeds.ravel()[border]==labelset[l])
        else:
            list_xm.append(0)
        
        Omega = np.ones(npixel)
        if anchor_weights is not None:
            Omega[anchor['imask']] = anchor_weights[l]
        list_Omega.append(wanchor * Omega[unknown])
        
        if ground_truth is not None:
            list_GT.append(
                ground_truth.ravel()[unknown]==labelset[l],
                )
      
    ## solve RW system 
    if not per_label:
        x = solve_at_once(
            Lu,B,list_xm,list_Omega,list_x0, list_GT=list_GT, **kwargs)
    else:
        x = solve_per_label(
            Lu,B,list_xm,list_Omega,list_x0,**kwargs)
        
    ## reshape solution
    y = (seeds.ravel()==np.c_[labelset]).astype(float)
    y[:,unknown] = x
    sol = labelset[np.argmax(y,axis=0)].reshape(image.shape)
    
    ## output arguments
    rargs = []
    for arg in return_arguments:
        if   arg=='image':  rargs.append(sol)
        elif arg=='y':      rargs.append(y)
        elif arg=='laplacian': rargs.append((L,border,B))
        else: pass
    
    if len(rargs)==1: return rargs[0]
    else: return tuple(rargs)
    
    
def solve_at_once(Lu,B,list_xm,list_Omega,list_x0, list_GT=[], **kwargs):
    ''' xm,Omega,x0 are lists of length nlabel'''
    
    nlabel = len(list_xm)
    
    ## intermediary matrices
    LL  = sparse.kron(np.eye(nlabel), Lu)
    BB  = sparse.kron(np.eye(nlabel), B)

    x0 = np.asmatrix(np.c_[list_x0].ravel()).T
    xm = np.asmatrix(np.c_[list_xm].ravel()).T
    Omega = sparse.spdiags(np.c_[list_Omega].ravel(), 0, *LL.shape)
                
                
    ## ground truth
    if list_GT!=[]:
        ''' Ax >= 0 '''
        N = Lu.shape[0]
        GT = np.asarray(list_GT).T
        S = np.cumsum(GT,axis=0)
        i_ = np.where(~GT)
        rows = (i_[1]-S[i_])*N + i_[0]
        cols =  i_[1]*N + i_[0]
        
        A = coo_matrix(
            (-np.ones(len(rows)), (rows,cols)),
            shape=(N*(nlabel-1),N*nlabel),
            ).tocsr()
        
        A = A + sparse.bmat(
            [[sparse.spdiags(list_gt[l],0,*Lu.shape) for l in range(label)] \
                for l2 in range(nlabel-1)]
            )
        import ipdb; ipdb.set_trace() ## to check ...
            
            
                
    ## solve
    optim_solver = kwargs.pop('optim_solver','unconstrained')
    logger.debug(
        'solve RW at once with solver="{}"'.format(optim_solver))
        
    P = LL + Omega
    q = BB*xm - Omega*x0

    if P.nnz==0:
        logger.warning('in QP, P=0. Returning 1-(q>0)') 
        x = (1 - (q>0))/(nlabel - 1)
    else:
        # if list_GT!=[]:
            # x = solve_qp_ground_truth(P,q,nlabel,x0,A) ## TODO 
        if optim_solver=='constrained':
            x = solve_qp_constrained(P,q,nlabel,x0)
        elif optim_solver=='unconstrained':
            x = solve_qp(P, q, **kwargs)
        else:
            raise Exception('Did not recognize solver: {}'.format(optim_solver))
    return x.reshape((nlabel,-1))
    
##------------------------------------------------------------------------------
def solve_per_label(Lu,B,list_xm,list_Omega,list_x0, **kwargs):
    ''' xm,Omega,x0 are lists of length nlabel'''
    
    nlabel = len(list_xm)

    ## compute separately for each label (best)
    nvar = Lu.shape[0]
    x = np.zeros((nlabel,nvar))

    ## solver
    optim_solver = kwargs.pop('optim_solver','unconstrained')
    logger.debug(
        'solve RW per label with solver="{}"'.format(optim_solver))
    
    ## compute tolerance depending on the Laplacian
    rtol = kwargs.pop('rtol', 1e-6)
    tol = np.maximum(np.max(Lu.data),np.max(list_Omega))*rtol
    logger.debug('absolute CG tolerance = {}'.format(tol))
    for s in range(nlabel - 1):## don't need to solve the last one !
        x0 = np.asmatrix(np.c_[list_x0[s]])
        xm = np.asmatrix(np.c_[list_xm[s]])

        Omega = sparse.spdiags(np.c_[list_Omega[s]].ravel(), 0, *Lu.shape)
                
        ## solve
        P = Lu + Omega
        q = B*xm - Omega*x0
        
        if P.nnz==0:
            logger.warning('in QP, P=0. Returning 1-(q>0)') 
            x = (1 - (q>0))/(nlabel - 1)
        else:
            if optim_solver=='constrained':
                raise Exception(
                    'Constrained solver not implemented for per-label method')
            elif optim_solver=='unconstrained':
                _x = solve_qp(P, q, tol=tol,**kwargs)
            else:
                raise Exception(
                    'Did not recognize solver: {}'.format(optim_solver))
        x[s] = _x.ravel()
    
    ## last label
    x[-1] = 1 - np.sum(x,axis=0)
    return x
        
##------------------------------------------------------------------------------
def solve_qp(P,q,**kwargs):
    import solver_qp as solver
    reload(solver)  ## TODO: tolerance bug ? (fails if tol < 1e-13)
    return solver.solve_qp(P,q,**kwargs)
    
    
##------------------------------------------------------------------------------
def solve_qp_constrained(P,q,nlabel,x0,**kwargs):
    import solver_qp_constrained as solver
    reload(solver)
    
    ## constrained solver
    nvar = q.size
    npixel = nvar/nlabel
    F = sparse.bmat([
        [sparse.bmat([[-sparse.eye(npixel,npixel) for i in range(nlabel-1)]])],
        [sparse.eye(npixel*(nlabel-1),npixel*(nlabel-1))],
        ])
        
    ## quadratic objective
    objective = solver.ObjectiveAPI(P, q, G=1, h=0, F=F,**kwargs)
    
    ## log barrier solver
    t0 = 1
    mu = 20
    epsilon = 1e-3
    solver = solver.ConstrainedSolver(
        objective,
        t0=t0,
        mu=mu,
        epsilon=epsilon,
        )
    
    ## remove zero entries in initial guess
    xinit = x0.reshape((-1,nlabel),order='F')
    xinit[xinit<1e-10] = 1e-3
    xinit = (xinit/np.c_[np.sum(xinit, axis=1)]).reshape((-1,1),order='F')
    
    x = solver.solve(xinit, **kwargs)
    return x
    

##------------------------------------------------------------------------------
def energy_anchor(
        image,
        x,
        anchor_api,
        seeds=[],
        # anchor_function=None,
        weight_function=None,
        **kwargs
        ):
    
    ## constants
    labelset    = anchor_api.get_labelset()
    nlabel      = len(labelset)
    npixel      = x[0].size
    inds        = np.arange(npixel)
    marked      = np.where(np.in1d(seeds,labelset))[0]
    unknown     = np.setdiff1d(inds,marked, assume_unique=True)
    nunknown    = unknown.size
    
    beta    = kwargs.pop('beta', 1.)
    
    ## anchor function:
    # D = compute_D(  
            # image, 
            # marked=marked, 
            # weight_function=weight_function,
            # beta=beta)
    # anchor, anchor_weights = anchor_api.get_anchor_and_weights(D)
    anchor, anchor_weights = anchor_api.get_anchor_and_weights(1)
    
    ## per label lists of vectors
    list_x0, list_Omega, list_xm = [],[],[]
    for l in range(nlabel):
        x0 = 1/float(nlabel) * np.ones(npixel)
        x0[anchor['imask']] = anchor['data'][l]
        list_x0.append(x0[unknown])
        
        Omega = np.ones(npixel)
        if anchor_weights is not None:
            Omega[anchor['imask']] = anchor_weights[l]
        list_Omega.append(Omega[unknown])
    
    
    en = 0
    for label in range(nlabel):
        xu = x[label][unknown]
        en += np.sum(list_Omega[label] * (xu - list_x0[label])**2)
        
    return float(en)
    
##------------------------------------------------------------------------------
def energy_rw(
        image,
        x,
        seeds=[],
        weight_function=None,
        **kwargs
        ):
        
    ## constants
    nlabel      = len(x)
    npixel      = image.size
    inds        = np.arange(npixel)
    labelset    = np.array(kwargs.pop('labelset', range(nlabel)), dtype=int)
    marked      = np.where(np.in1d(seeds,labelset))[0]
    unknown     = np.setdiff1d(inds,marked, assume_unique=True)
    nunknown    = unknown.size
    
    beta    = kwargs.pop('beta', 1.)
    
    ## generate segmentation vector
    ## compute laplacian
    logger.debug('compute laplacian')
    L, border, B = compute_laplacian(
        image,
        marked=marked,
        beta=beta, 
        weight_function=weight_function,
        )
        
    ## compute energy
    en = 0.0
    for label in range(nlabel):
        X = np.asmatrix(x[label][unknown]).T
        en += X.T * L * X
    return float(en)
    
##------------------------------------------------------------------------------
def compute_D(
        image, 
        marked=None,
        beta=1., 
        weight_function=None,
        ):
    im = np.asarray(image)
    N  = im.size
    
    ## compute weights
    logger.debug('compute D')
    if weight_function is None:
        # if no wf provided, use standard with provided beta
        weight_function = lambda x: weight_std(x,beta=beta)
    ij, data = weight_function(im)
    
    ## affinity matrix
    A = sparse.coo_matrix((data, ij), shape=(N,N))
    A = A + A.T
    
    D = np.asarray(A.sum(axis=0)).ravel()
    
    if marked is None:
        return D
    else:
        unknown = np.setdiff1d(np.arange(im.size), marked, assume_unique=True)
        D[unknown]
        return D
    
##------------------------------------------------------------------------------
def compute_laplacian(
        image, 
        marked=None,
        beta=1., 
        weight_function=None,
        return_D=False,
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
    logger.debug('compute L weights')
    if weight_function is None:
        # if no wf provided, use standard with provided beta
        weight_function = lambda x: weight_std(x,beta=beta)
    ij, data = weight_function(im)
    
    ## affinity matrix
    logger.debug('laplacian matrix')
    A = sparse.coo_matrix((data, ij), shape=(N,N))
    A = A + A.T
    
    ## laplacian matrix
    L = (sparse.spdiags(A.sum(axis=0),0,N,N) - A).tocsr()
    
    ## return laplacian
    if marked is None and return_D:
        D = A.sum(axis=0)
        return L,D
    elif marked is None:
        return L
        
    ## if marked is not None,
    ## keep only unknown pixels, and marked pixels touching unknown
    logger.debug('decompose into Lu, B')
    unknown = np.setdiff1d(np.arange(im.size), marked, assume_unique=True)
    
    mask = np.ones(im.shape, dtype=int)
    mask.flat[unknown] = 0
    ijborder = (mask.flat[ij[0]] + mask.flat[ij[1]])==1
    mask.flat[ij[0][ijborder]] *= 2
    mask.flat[ij[1][ijborder]] *= 2
    border = np.where(mask.ravel()>=2)[0]

    Lu = L[unknown,:][:,unknown]
    B  = L[unknown,:][:,border]
    
    if return_D:
        D = np.asarray(A.sum(axis=0)).flat[unknown]
        return Lu,border,B,D
    else:
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

