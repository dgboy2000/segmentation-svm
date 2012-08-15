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
logger = utils_logging.get_logger('rwsegment',utils_logging.DEBUG)
        
##------------------------------------------------------------------------------
def segment(
        image,
        prior,
        seeds=[],
        prior_function=None,
        loss=None,
        weight_function=None,
        return_arguments=['image'],
        **kwargs
        ):
    ''' segment an image with method from MICCAI'12
    
    arguments:
        image   = image to segment
        prior   = a list of vectors of size npixel
                
    keyword arguments:
        seeds    = label map, with -1 for unmarked pixels
                 must have the same shape as image
        labelset = output labels (in the order of prior)
                 seed labels not in labelset are ignored
                 
        prior_weights = a list of vectors of size npixel
        loss = a list of vectors of size npixel
        
        beta    = contrast parameter if no weight_function is provided
        weight_function = user provided function:
            (ij, data) = weight_function(image)
        add 'laplacian' to return_arguments
        
                
    '''
    
    ## constants
    # assume prior has shape (nunknown, nlabel)
    nlabel      = len(prior['mean']) 
    npixel      = image.size
    labelset    = np.array(kwargs.pop('labelset', range(nlabel)), dtype=int)
    inds        = np.arange(npixel)
    marked      = np.where(np.in1d(seeds,labelset))[0]
    unknown     = np.setdiff1d(inds,marked, assume_unique=True)
    nunknown    = unknown.size
        
    ## parameters
    beta    = kwargs.pop('beta', 1.)
    wprior   = kwargs.pop('wprior', 1.)
    per_label = kwargs.pop('per_label',True)

    ## compute laplacian
    logger.debug('compute laplacian')
        
    L, border, B,D = compute_laplacian(
        image,
        marked=marked,
        beta=beta, 
        weight_function=weight_function,
        return_D=True,
        )

    ## prior function:
    prior_weights = None
    if prior_function is not None:
        prior_weights = prior_function(D)
        
    ## per label lists of vectors
    list_x0, list_Delta, list_xm, list_z = [],[],[],[]
    for l in range(nlabel):
        x0 = 1/float(nlabel) * np.ones(npixel)
        x0[prior['imask']] = prior['mean'][l]
        list_x0.append(x0[unknown])
        if seeds!=[]:
            list_xm.append(seeds.ravel()[border]==l)
        else:
            list_xm.append(0)
        if loss is not None:
            z = np.asarray(loss[l])
        else:
            z = np.zeros(npixel)
        list_z.append(z[unknown])
        
        Delta = np.ones(npixel)
        if prior_weights is not None:
            Delta[prior['imask']] = prior_weights[l]
        list_Delta.append(wprior * Delta[unknown])
      
    ## solve RW system 
    if not per_label:
        x = solve_at_once(
            L,B,list_xm,list_Delta,list_x0,list_z,**kwargs)
    else:
        x = solve_per_label(
            L,B,list_xm,list_Delta,list_x0,list_z,**kwargs)
        
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
    
    
def solve_at_once(Lu,B,list_xm,list_Delta,list_x0,list_z, **kwargs):
    ''' xm,Delta,x0 and z are lists of length nlabel'''
    
    nlabel = len(list_xm)
    
    ## intermediary matrices
    LL  = sparse.kron(np.eye(nlabel), Lu)
    BB  = sparse.kron(np.eye(nlabel), B)

    x0 = np.asmatrix(np.c_[list_x0].ravel()).T
    xm = np.asmatrix(np.c_[list_xm].ravel()).T
    if list_z is not None:
        z = np.asmatrix(np.c_[list_z].ravel()).T
    else:
        z = 0
        
    Delta = sparse.spdiags(np.c_[list_Delta].ravel(), 0, *LL.shape)
                
    ## solve
    optim_solver = kwargs.pop('optim_solver','unconstrained')
    logger.debug(
        'solve RW at once with solver="{}"'.format(optim_solver))
        
    P = LL + Delta
    q = BB*xm - Delta*x0 + z

    if P.nnz==0:
        logger.warning('in QP, P=0. Returning 1-(q>0)') 
        x = (1 - (q>0))/(nlabel - 1)
    else:
        if optim_solver=='constrained':
            x = solve_qp_constrained(P,q,nlabel,x0)
        elif optim_solver=='unconstrained':
            x = solve_qp(P, q, **kwargs)
        else:
            raise Exception('Did not recognize solver: {}'.format(optim_solver))
    return x.reshape((nlabel,-1))
    
##------------------------------------------------------------------------------
def solve_per_label(Lu,B,list_xm,list_Delta,list_x0,list_z, **kwargs):
    ''' xm,Delta,x0 and z are lists of length nlabel'''
    
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
    tol = np.maximum(np.max(Lu.data),np.max(list_Delta))*rtol
    
    for s in range(nlabel - 1):## don't need to solve the last one !
        x0 = np.asmatrix(np.c_[list_x0[s]])
        xm = np.asmatrix(np.c_[list_xm[s]])
        
        if list_z is not None:
            z = np.asmatrix(np.c_[list_z[s]])
        else:
            z = 0
            
        Delta = sparse.spdiags(np.c_[list_Delta[s]].ravel(), 0, *Lu.shape)
                
        ## solve
        P = Lu + Delta
        q = B*xm - Delta*x0 + z

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
    reload(solver)
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
def energy_prior(
        image,
        x,
        prior,
        seeds=[],
        prior_function=None,
        weight_function=None,
        **kwargs
        ):
    
    ## constants
    # assume prior has shape (nunknown, nlabel)
    nlabel      = len(prior['mean']) 
    npixel      = x[0].size
    labelset    = np.array(kwargs.pop('labelset', range(nlabel)), dtype=int)
    inds        = np.arange(npixel)
    marked      = np.where(np.in1d(seeds,labelset))[0]
    unknown     = np.setdiff1d(inds,marked, assume_unique=True)
    nunknown    = unknown.size
    
    beta    = kwargs.pop('beta', 1.)
    
    ## prior function:
    prior_weights = None
    if prior_function is not None:
        D = compute_D(  
            image, 
            marked=marked, 
            weight_function=weight_function,
            beta=beta)
        prior_weights = prior_function(D)
    
    ## per label lists of vectors
    list_x0, list_Delta, list_xm, list_z = [],[],[],[]
    for l in range(nlabel):
        x0 = 1/float(nlabel) * np.ones(npixel)
        x0[prior['imask']] = prior['mean'][l]
        list_x0.append(x0[unknown])
        
        Delta = np.ones(npixel)
        if prior_weights is not None:
            Delta[prior['imask']] = prior_weights[l]
        list_Delta.append(Delta[unknown])
    
    
    en = 0
    for label in range(nlabel):
        xu = x[label][unknown]
        en += np.sum(list_Delta[label] * (xu - list_x0[label])**2)
        
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
    # assume prior has shape (nunknown, nlabel)
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

