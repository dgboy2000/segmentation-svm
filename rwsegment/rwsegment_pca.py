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
    pca = anchor_api.prior['eigenvectors'] 
 
        
    ## per label lists of vectors
    list_x0, list_Omega, list_xm = [],[],[]
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
        
      
    ## solve RW system
    x,coefs = solve_at_once(
        Lu,B,list_xm,list_Omega,list_x0, pca,
        **kwargs)

    ppca = np.asarray(list_x0) + np.sum(pca*coefs,axis=1).reshape((nlabel,-1))

    ## reshape solution
    y = (seeds.ravel()==np.c_[labelset]).astype(float)
    y[:,unknown] = x + ppca
    sol = labelset[np.argmax(y,axis=0)].reshape(image.shape)
    
    solpca = np.zeros(image.shape)
    solpca.flat[unknown] = labelset[np.argmax(ppca,axis=0)]

    ## output arguments
    rargs = []
    for arg in return_arguments:
        if   arg=='image':  rargs.append(sol)
        elif arg=='y':      rargs.append(y)
        elif arg=='laplacian': rargs.append((L,border,B))
        elif arg=='pca': rargs.append(coefs)
        elif arg=='impca': rargs.append(solpca)
        else: pass
    
    if len(rargs)==1: return rargs[0]
    else: return tuple(rargs)
    
    
def solve_at_once(Lu,B,list_xm,list_Omega,list_x0, pca, **kwargs):
    ''' xm,Omega,x0 are lists of length nlabel'''
    
    nlabel = len(list_xm)

    ## intermediary matrices
    LL  = sparse.kron(np.eye(nlabel), Lu)
    BB  = sparse.kron(np.eye(nlabel), B)

    x0 = np.asmatrix(np.c_[list_x0].ravel()).T
    xm = np.asmatrix(np.c_[list_xm].ravel()).T
    Omega = sparse.spdiags(np.c_[list_Omega].ravel(), 0, *LL.shape)

    U = sparse.coo_matrix(pca)
    npca = U.shape[1]

    ## solve
    optim_solver = kwargs.pop('optim_solver','unconstrained')
    
    P = sparse.bmat([[LL + Omega, LL*U],[U.T*LL.T, U.T*U]])
    q = sparse.bmat([[sparse.eye(*LL.shape)],[U.T]]) * (BB*xm + LL*x0)
    c = 0
    
    ## compute tolerance depending on the Laplacian
    rtol = kwargs.pop('rtol', 1e-6)
    tol = np.max(P.data)*rtol
    logger.debug('absolute CG tolerance = {}'.format(tol))

    if optim_solver=='constrained':
        ## probability distribution constrained
        logger.debug(
            'solve RW at once with constrained solver')
        x = solve_qp_constrained(P,q,nlabel,x0,c=c,**kwargs)
    elif optim_solver=='unconstrained':
        ## unconstrained
        logger.debug(
            'solve RW at once with unconstrained solver')
        x = solve_qp(P, q, tol=tol,**kwargs)
    else:
        raise Exception('Did not recognize solver: {}'.format(optim_solver))


    logger.debug('pca coefs: {}'.format( x.ravel()[-npca:]))

    return x[:-npca].reshape((nlabel,-1)), x[-npca:]
    
#------------------------------------------------------------------------------
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
    t0 = kwargs.pop('logbarrier_initial_t',1.0)
    mu = kwargs.pop('logbarrier_mu',20.0)
    epsilon = kwargs.pop('logbarrier_epsilon',1e-3)
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
    # import ipdb; ipdb.set_trace()
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

