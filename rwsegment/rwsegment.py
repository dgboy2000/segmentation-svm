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
logger = utils_logging.get_logger('rwsegment',utils_logging.INFO)
        
class BaseAnchorAPI(object):
    def __init__(self,anchor, anchor_weight=1.0, **kwargs):
        self.anchor = anchor
        self.labelset = anchor['labelset']
        self.imask = anchor['imask']
        self.anchor_weight = anchor_weight
        
    def get_labelset(self):
        return self.labelset
    
    def get_anchor_and_weights(self, list_D, indices):
        nlabel = len(self.labelset)
        N = np.maximum(np.max(self.imask), np.max(indices))+ 1
        omega = 1./nlabel * np.ones((nlabel,N))
        omega[:,self.imask] = self.anchor['data']
        omega = omega[:,indices]
        weights = self.anchor_weight * np.ones(omega.shape) * np.asarray(list_D)
        return omega, weights
        
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
   
    if len(unknown)==0:
        logger.error('no unknown pixels in image')
        sys.exit()

    ## parameters
    beta    = kwargs.pop('beta', 1.)
    per_label = kwargs.pop('per_label',True)
    
    ## ground truth
    ground_truth = kwargs.pop('ground_truth',None)
    ground_truth_init = kwargs.pop('ground_truth_init',None)
       
    ## seeds values 
    seeds_prob = kwargs.pop('seeds_prob', None)
    
    ## compute laplacian
    logger.debug('compute laplacian')
    list_Lu, list_B, list_D, border = compute_laplacians(
        image,
        marked=marked,
        beta=beta, 
        weight_function=weight_function,
        )
    if len(list_Lu)==1:
        list_Lu = [list_Lu[0] for i in range(nlabel)]
        list_B  = [list_B[0]  for i in range(nlabel)]
        list_D  = [list_D[0]  for i in range(nlabel)]
        
    ## anchor function
    list_x0, list_Omega = anchor_api.get_anchor_and_weights(list_D, unknown)
       
    ## per label lists of vectors
    list_xm, list_GT, list_GT_init = [],[],[]
    for l in range(nlabel):
        
        if seeds_prob is not None:
            list_xm.append(seeds_prob[l][border])
        elif seeds!=[]:
            list_xm.append(seeds.ravel()[border]==labelset[l])
        else:
            list_xm.append(0)
        
        ## ground truth
        if ground_truth is not None:
            list_GT.append(ground_truth[l][unknown])
        if ground_truth_init is not None:
            list_GT_init.append(ground_truth_init[l][unknown])
      
    ## if additional laplacian
    addL = kwargs.pop('additional_laplacian', None)
    if addL is not None:
        logger.debug('additional laplacian in solve at once')
        uunknown = (unknown + np.c_[np.arange(nlabel)*npixel]).ravel()
        addL = addL[uunknown,:][:,uunknown]
        if per_label is True:
            logger.warning('additional lapacian is not None, using "solve at once"')
            per_label = False

    ## if additional linear term
    addlin = kwargs.pop('additional_linear',None)
    if addlin is not None:
        addq = np.mat(np.asarray(addlin)[:,unknown].ravel()).T
        if per_label is True:
            logger.warning('additional linear is not None, using "solve at once"')
            per_label = False
    else:
        addq = None

    ## solve RW system
    if ground_truth is not None or not per_label:
        x = solve_at_once(
            list_Lu, list_B, list_xm, list_Omega, list_x0, 
            list_GT=list_GT, list_GT_init=list_GT_init, 
            additional_laplacian=addL,
            additional_linear=addq,
            **kwargs)
    else:
        x = solve_per_label(
            list_Lu,list_B,list_xm,list_Omega,list_x0,**kwargs)
    
    ## normalize x
    x = np.clip(x, 0,1)
    x = x / np.sum(x, axis=0) 

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
    
    
def solve_at_once(list_Lu, list_B, list_xm, list_Omega, list_x0, list_GT=[], **kwargs):
    ''' xm,Omega,x0 are lists of length nlabel'''
    
    nlabel = len(list_xm)
    
    ## intermediary matrices
    #if len(list_Lu)==1:
    #    LL  = sparse.kron(np.eye(nlabel), list_Lu[0])
    #    BB  = sparse.kron(np.eye(nlabel), list_B[0])
    #else:
    LL = sparse.bmat([[sparse.kron(np.arange(nlabel)==i,Lu)] for i,Lu in enumerate(list_Lu)])
    BB = sparse.bmat([[sparse.kron(np.arange(nlabel)==i,B)]  for i,Lu in enumerate(list_B)])

    x0 = np.asmatrix(np.c_[list_x0].ravel()).T
    xm = np.asmatrix(np.c_[list_xm].ravel()).T
    Omega = sparse.spdiags(np.c_[list_Omega].ravel(), 0, *LL.shape)

    addL = kwargs.pop('additional_laplacian',None)
    if addL is None:
        addL = sparse.csr_matrix(LL.shape)

    addlin = kwargs.pop('additional_linear', None)
    if addlin is None:
        addq = 0 
    else:
        addq = addlin

    ## solve
    optim_solver = kwargs.pop('optim_solver','unconstrained')
    
    P = LL + Omega + addL
    q = BB*xm - Omega*x0 + addq
    # c = x0.T*Omega*x0
    c = 0
    
    if P.nnz==0:
        # if no laplacian 
        logger.warning('in QP, P=0. Returning 1-(q>0)') 
        x = (1 - (q>0))/float(nlabel - 1)
    elif np.sqrt(np.dot(q.T,q)) < 1e-10:
        # if no linear term: x is constant and has to remain a probability
        x = 1./nlabel * np.mat(np.ones(q.shape))
    else:
        ## compute tolerance depending on the Laplacian
        rtol = kwargs.pop('rtol', 1e-6)
        tol = np.max(P.data)*rtol
        logger.debug('absolute CG tolerance = {}'.format(tol))

        if list_GT!=[]:
            ## ground truth constrained
            logger.debug(
                'solve RW at once with ground-truth constrained solver')
            x = solve_qp_ground_truth(P,q,list_GT,nlabel,c=c,**kwargs) 
        elif addlin is not None:
            ## added linear term: use contrained optim
            logger.debug(
                'solve RW at once with constrained solver (added linear term)')
            x = solve_qp_constrained(P,q,nlabel,x0,c=c,**kwargs)
        elif optim_solver=='constrained':
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
    return x.reshape((nlabel,-1))
    
##------------------------------------------------------------------------------
def solve_per_label(list_Lu, list_B, list_xm, list_Omega, list_x0, **kwargs):
    ''' xm,Omega,x0 are lists of length nlabel'''
    nlabel = len(list_xm)
    
    ## compute separately for each label (best)
    nvar = list_Lu[0].shape[0]
    x = np.zeros((nlabel,nvar))

    ## solver
    optim_solver = kwargs.pop('optim_solver','unconstrained')
    logger.debug(
        'solve RW per label with solver="{}"'.format(optim_solver))
    
    ## compute tolerance depending on the Laplacian
    rtol = kwargs.pop('rtol', 1e-6)
    
    for s in range(nlabel - 1):## don't need to solve the last one !
        Lu = list_Lu[s]
        B  = list_B[s]
        
        ## prior
        Omega = sparse.spdiags(np.c_[list_Omega[s]].ravel(), 0, *Lu.shape)
        x0 = np.asmatrix(np.c_[list_x0[s]])
        
        ## if no laplacian
        if len(Lu.data)==0 or np.max(np.abs(Lu.data))<1e-10:
            x[s] = x0.ravel()
            
        ## set tolerance
        tol = np.maximum(np.max(Lu.data),np.max(list_Omega[s]))*rtol
        logger.debug('absolute CG tolerance = {}'.format(tol))
        
        ## seeds
        xm = np.asmatrix(np.c_[list_xm[s]].astype(float))
        
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
def solve_qp_ground_truth(P,q,list_GT,nlabel,**kwargs):
   
    GT = np.asarray(list_GT).T
    
    ''' Ax >= b '''
    N = P.shape[0]/nlabel
    S = np.cumsum(GT,axis=1)
    i_ = np.where(~GT)
    rows = (i_[1]-S[i_])*N + i_[0]
    cols =  i_[1]*N + i_[0]
    
    G = sparse.coo_matrix(
        (-np.ones(len(rows)), (rows,cols)),
        shape=(N*(nlabel-1),N*nlabel),
        ).tocsr()
    
    G = G + sparse.bmat(
        [[sparse.spdiags(list_GT[l],0,N,N) for l in range(nlabel)] \
            for l2 in range(nlabel-1)]
        )
    h = kwargs.pop('ground_truth_margin',0.0)

    n = q.size
    ## positivity constraint
    G = sparse.bmat([[G], [sparse.eye(n,n)]])

   
    npixel = n/nlabel
    F = sparse.bmat([
        [sparse.bmat([[-sparse.eye(npixel,npixel) for i in range(nlabel-1)]])],
        [sparse.eye(npixel*(nlabel-1),npixel*(nlabel-1))],
        ])

    ## initial guess
    list_GT_init = kwargs.pop('list_GT_init',None)
    if list_GT_init is None:
        xinit = np.asmatrix(np.asarray(list_GT).ravel()).T
    else:
        xinit = np.asmatrix(np.asarray(list_GT_init).ravel()).T
 
    use_mosek = kwargs.pop('use_mosek', True)
    if use_mosek:
        import solver_mosek as solver         
        reload(solver)
 
        logger.info('use mosek')
        objective = solver.ObjectiveAPI(P, q, G=G, h=h,F=F,**kwargs)
        constsolver = solver.ConstrainedSolver(
            objective,
            )

        p = xinit.A.reshape((nlabel,-1)) + 1e-8
        xinit = np.mat((p/np.sum(p,axis=0)).reshape(xinit.shape))
 
        x = constsolver.solve(xinit) 

        minx = np.min(x)
        maxx = np.max(x)
        logger.info('Done solver ground truth constrained: x in [{:.3}, {:.3}]'.format(minx,maxx))
        logger.info('clipping x in [0,1]')
        prob = np.clip(x.reshape((nlabel,-1)), 0,1)
        prob = prob / np.sum(prob,axis=0)
        nx = prob.reshape(x.shape)
       
        #check validity
        ndiff = np.sum(np.argmax(prob, axis=0)!=np.argmax(list_GT,axis=0))
        logger.info('number incorrect pixels: {}'.format(ndiff))
        return nx


    ## else: log barrier    
    import solver_qp_constrained as solver
    reload(solver)
    
    ## quadratic objective
    #objective = solver.ObjectiveAPI(P, q, G=G, h=h,**kwargs)
    objective = solver.ObjectiveAPI(P, q, G=G, h=h,F=F,**kwargs)
    
    ## log barrier solver
    t0      = kwargs.pop('logbarrier_initial_t',1.0)
    mu      = kwargs.pop('logbarrier_mu',20.0)
    epsilon = kwargs.pop('logbarrier_epsilon',1e-4)
    modified = kwargs.pop('logbarrier_modified',False)
    maxiter = kwargs.pop('logbarrier_maxiter', 10)
    solver = solver.ConstrainedSolver(
        objective,
        t0=t0,
        mu=mu,
        epsilon=epsilon,
        modified=modified,
        )
    
    ## internal solver is newton's method
    newton_a       = kwargs.pop('newton_a', 0.4)
    newton_b       = kwargs.pop('newton_b', 0.8)
    newton_epsilon = kwargs.pop('newton_epsilon', 1e-4)
    newton_maxiter = kwargs.pop('newton_maxiter', 100)
   
    p = xinit.A.reshape((nlabel,-1)) + 1e-8
    xinit = np.mat((p/np.sum(p,axis=0)).reshape(xinit.shape))
 
    x = solver.solve(
        xinit, 
        a=newton_a,
        b=newton_b,
        epsilon=newton_epsilon,
        maxiter=newton_maxiter,
        )
    
    minx = np.min(x)
    maxx = np.max(x)
    logger.info('Done solver ground truth constrained: x in [{:.3}, {:.3}]'.format(minx,maxx))
    logger.info('clipping x in [0,1]')
    prob = np.clip(x.reshape((nlabel,-1)), 0,1)
    prob = prob / np.sum(prob,axis=0)
    nx = prob.reshape(x.shape)
    return nx

    
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
def energy_anchor(
        image,
        x,
        anchor_api,
        seeds=[],
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
    beta        = kwargs.pop('beta', 1.)
    
    ## anchor function
    logger.debug('compute laplacian')
    list_Lu, list_B, list_D, border = compute_laplacians(
        image,
        marked=marked,
        beta=beta, 
        weight_function=weight_function,
        )
    if len(list_D)==1:
        list_D  = [list_D[0]  for i in range(nlabel)]
        
    #import ipdb; ipdb.set_trace()
    list_x0, list_Omega = anchor_api.get_anchor_and_weights(list_D, unknown) ## D ?
    
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
    beta        = kwargs.pop('beta', 1.)
    
    ## compute laplacian
    logger.debug('compute laplacian')
    list_Lu, list_B, list_D, border = compute_laplacians(
        image,
        marked=marked,
        beta=beta, 
        weight_function=weight_function,
        )
    if len(list_Lu)==1:
        list_Lu = [list_Lu[0] for i in range(nlabel)]
        list_B  = [list_B[0]  for i in range(nlabel)]
    
    '''
    try:
        L, border, B, D = compute_laplacian(
            image,
            marked=marked,
            beta=beta, 
            weight_function=weight_function,
            )
    except MemoryError as e:
        import gc
        import mpi
        gc.collect()
        logger.error(
            'Memory error computing Laplacian in process #{}'\
            .format(mpi.RANK))
        L, border, B, D = compute_laplacian(
            image,
            marked=marked,
            beta=beta, 
            weight_function=weight_function,
            )
    '''    
        
    ## compute energy
    en = 0.0
    for label in range(nlabel):
        X = np.asmatrix(x[label][unknown]).T
        en += X.T * list_Lu[label] * X
        
        ## seeds !!
        xm = seeds.ravel()[border]==labelset[label]
        en += X.T * list_B[label] * np.mat(xm.reshape((-1,1)))
    return float(en)

    
##------------------------------------------------------------------------------
def compute_laplacians(
        image, 
        marked=None,
        beta=1., 
        weight_function=None,
        ):
    ''' compute laplacian matrix for using with Random Walks
    args:
        image
        marked  : indices of marked pixels (in image.flat)
        beta    : contrast parameter for default weight_function:
            for touching pixel pair (i,j),
            wij = exp (- beta (image.flat[i] - image.flat[j])^2)
           
        weight_function : user function to compute edge weights
    '''
    
    im = np.asarray(image)
    npix = im.size
    inds = np.arange(im.size).reshape(im.shape)

    all_i = np.r_[tuple([inds.take(np.arange(im.shape[d]-1), axis=d).ravel() \
            for d in range(im.ndim)])]
    all_j = np.r_[tuple([inds.take(np.arange(1,im.shape[d]), axis=d).ravel() \
            for d in range(im.ndim)])]

    ## select only unknown pairs
    if marked is not None:
        is_unknown = np.ones(npix, dtype=bool)
        is_unknown[marked] = False
        is_unknown_i = is_unknown[all_i]
        is_unknown_j = is_unknown[all_j]
        is_unknown_p = np.logical_or(is_unknown_i, is_unknown_j)
        i = all_i[is_unknown_p]
        j = all_j[is_unknown_p]
        
        unknown = np.where(is_unknown)[0]
        is_border_p = np.logical_xor(is_unknown_i, is_unknown_j)
        is_border_i = np.logical_and(np.logical_not(is_unknown_i), is_border_p)
        is_border_j = np.logical_and(np.logical_not(is_unknown_j), is_border_p)
        border = np.union1d(all_i[is_border_i], all_j[is_border_j])        
    else:
        i = all_i
        j = all_j
        
    ## compute weights
    logger.debug('compute L weights')
    if weight_function is None:
        # if no wf provided, use standard with provided beta
        weight_function = lambda x,i,j: weight_std(x,i,j,beta=beta)
        
    ## weight function may depend on label
    list_Lu, list_B, list_D = [], [], []
    for data in  weight_function(im,i,j):
        ## affinity matrix
        logger.debug('laplacian matrix')
        A = sparse.coo_matrix((data, (i,j)), shape=(npix,npix))
        A = A + A.T
        
        ## laplacian matrix
        L = (sparse.spdiags(A.sum(axis=0),0,npix,npix) - A).tocsr()
        
        ## return laplacian
        if marked is None:
            Lu = L 
            B  = sparse.coo_matrix((0,0))
            D  = np.asarray(A.sum(axis=0))
        else:
            Lu = L[unknown,:][:,unknown]
            B  = L[unknown,:][:,border]
            D = np.asarray(A.sum(axis=0)).flat[unknown]
        
        list_Lu.append(Lu)
        list_B.append(B)
        list_D.append(D)
        
    return list_Lu, list_B, list_D, border
    
##------------------------------------------------------------------------------
def weight_std(image, i, j, beta=1.0):
    ''' standard weight function 
    
        for touching pixel pair (i,j),
            wij = exp (- beta (image.flat[i] - image.flat[j])^2)
    '''
    im = np.asarray(image)
    wij = np.exp(-beta * (image.flat[i] - image.flat[j])**2)
    return [wij]
    
