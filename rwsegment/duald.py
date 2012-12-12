'''
Created on Dec 6, 2012

@author: puneet
'''
import sys
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import solver_mosek as solver
reload(solver)
import svm_worker
import utils_logging
logger = utils_logging.get_logger('duald',utils_logging.INFO)

if '--parallel' in sys.argv:
    is_parallel = True
    from mpi4py import MPI
    Comm = MPI.COMM_WORLD
    MPIRANK = Comm.Get_rank()
    MPISIZE = Comm.Get_size()
else:
    is_parallel = False
    Comm = None
    MPIRANK = 0
    MPISIZE = 1



##------------------------------------------------------------------------------
def decompose_brute_force(n,nlabel):
    k = 2     
    counter = 0
    nsub = (n*(n-1))/2
    subproblems = []
    for i in range(n):
        for j in range(i+1, n):
            sub = []
            for k in range(0,nlabel*n,n):
                sub.extend([i+k,j+k])
            subproblems.append(sub)
    publicvars = range(n)
    return subproblems, publicvars
    
##------------------------------------------------------------------------------
def decompose_with_image_connectivity(shape, nlabel, size_sub=3, marked=[]):
    nunknown = np.prod(shape) - len(marked)
    ndim = len(shape)
    sh = np.asarray(shape)
    si = np.ones(ndim, dtype=int) * size_sub
    nsub = np.ceil((sh-1)/(si-1).astype(float))
    
    isunknown = np.ones(np.prod(shape), dtype=int)
    isunknown[marked] = 0
    indices = (np.cumsum(isunknown) - 1).reshape(shape)
    indices.flat[marked] = -1
    subproblems = []
    #publicvars = []
    for i in np.argwhere(np.ones(nsub)):
        ind = i*(si-1)
        slices = tuple([slice(ind[d],ind[d]+si[d]) for d in range(ndim)])
        sub = indices[slices].ravel()
        sub = sub[sub>=0]
        sub_bar = sub.tolist()
        for s in range(1,nlabel):
            sub_bar.extend(sub + s*nunknown)
        # for sub2 in subproblems:
            # inter = np.intersect1d(sub_bar, sub2, assume_unique=True)
            # publicvars.extend(inter.tolist())
        if len(sub_bar)>0:
            subproblems.append(sub_bar)
    #publicvars = np.unique(publicvars)
    
    #return subproblems, publicvars
    return subproblems
   

##------------------------------------------------------------------------------
class Worker(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
        while 1:
            #print 'waiting to receive data (rank={})'.format(MPIRANK)
            data = Comm.scatter(None, root=0)
            if len(data)==0:
                break
            rdata = solve_list_subproblems(data)
            #print 'processing subproblems data (rank={})'.format(MPIRANK)
            # rdata = []
            # for subp in data:
                # info, nlabel, Pk_bar, qk_bar, gtk_bar = subp
                ##print '    processing subproblem {} (rank={})'.format(info, MPIRANK)
                # xk = solver_gt(nlabel, Pk_bar, qk_bar, gtk_bar, **kwargs)
                # par_dual = float(0.5*xk.T*Pk_bar*xk + xk.T*qk_bar)
                # rdata.append((info,xk,par_dual))
            #print 'sending back data (rank={})'.format(MPIRANK)
            Comm.gather(rdata,root=0)
           
def solve_list_subproblems(subp_list, **kwargs):
    rdata = []
    for subp in subp_list:
        info, nlabel, Pk_bar, qk_bar, gtk_bar = subp
        xk = solver_gt(nlabel, Pk_bar, qk_bar, gtk_bar, **kwargs)
        par_dual = float(0.5*xk.T*Pk_bar*xk + xk.T*qk_bar)
        rdata.append((info,xk,par_dual))
    return rdata
    
##------------------------------------------------------------------------------
def dd_solver_gt(nlabel, Lu, q_bar, gt_bar, subproblems, D=0, **kwargs):
    niter = kwargs.pop('niter', 100)
    gamma = kwargs.pop('gamma', 1e1)#0.01)
    epsilon = kwargs.pop('epsilon', 1e-5)
    nsub = len(subproblems)
    nvar = q_bar.size
    
    ## runs solver on each sub problem
    npixel = nvar/nlabel
    
    # take lower diagonal elements of L
    Lu_diag = Lu[np.arange(npixel),np.arange(npixel)].A.ravel()
    L = Lu - sparse.spdiags(Lu_diag, 0, npixel, npixel)
    L = sparse.spdiags(-L.sum(axis=0).A.ravel(), 0, npixel, npixel) + L
    ie, je = L.nonzero()
    Delta = D + (Lu - L)[np.arange(npixel),np.arange(npixel)].A.ravel()
    
    P_bar = sparse.kron(np.eye(nlabel),L) + \
        sparse.spdiags(np.tile(Delta, nlabel), 0, nvar,nvar)
    

    # find number of times a node or an edge appears
    N = np.zeros(npixel, dtype=int)
    for pb_bar in subproblems:
        pb = pb_bar[:len(pb_bar)/nlabel]
        N += np.in1d(np.arange(npixel), pb)
    
    subproblems_matrices = []
    for ipb, pb_bar in enumerate(subproblems):
        
        ## decompose quadratic term
        pb = np.asarray(pb_bar[:len(pb_bar)/nlabel])
        npb = len(pb)
        
        Pk = L[pb,:][:,pb]
        Pk = Pk - sparse.spdiags(Pk[np.arange(npb), np.arange(npb)], 0, npb, npb)
        
        ## remove some nodes
        inodes = np.where(np.abs(Pk.sum(axis=1).A.ravel() + Delta[pb])>1e-8)[0]
        rnodes = np.setdiff1d(pb, pb[inodes], assume_unique=True)
        Pk = Pk[inodes,:][:,inodes]
        pb = pb[inodes]
        N[rnodes] -= 1
        npb = len(pb)
        pb_bar = (pb + np.c_[np.arange(0,nvar,npixel)]).ravel()
        
        ## add diagonal element
        Pk = Pk + sparse.spdiags(-Pk.sum(axis=0).A.ravel(), 0, npb, npb)
        Pk = Pk + sparse.spdiags(Delta[pb]/N[pb], 0, npb, npb)
        
        ## extend Pk
        Pk_bar = sparse.kron(np.eye(nlabel), Pk)
        
        ## decompose linear term
        n_bar = np.tile(N[pb], nlabel)
        qk_bar = np.mat(q_bar[pb_bar].A / np.c_[n_bar])
        
        ## ground truth 
        gtk_bar = gt_bar[pb_bar]
        
        ## store subproblems
        subproblems_matrices.append((pb_bar, Pk_bar, qk_bar, gtk_bar))
        
        #import ipdb; ipdb.set_trace()
        ## remove used edges 
        iek = np.where(
            np.in1d(ie, pb)&\
            np.in1d(je, pb)&
            (ie!=je))[0]        
        L.data[iek] = 0
        
    N_bar = np.tile(N, nlabel)
       
    ## check subproblems sum to orignal problem
    xx = np.mat(np.random.random(nvar)).T
    obj_0 = float(0.5 * xx.T*P_bar*xx + xx.T*q_bar)
    obj_d = 0.0
    for k, subp in enumerate(subproblems_matrices):
        sub, Pk_bar, qk_bar, gtk_bar = subp
        obj_d += float(0.5 * xx[sub].T*Pk_bar*xx[sub] + xx[sub].T*qk_bar)
    if np.abs(obj_0-obj_d)>1e-5:
        print 'objective (original) = {}'.format(obj_0)
        print 'objective (DD) = {}'.format(obj_d)
        1/0
        #import ipdb; ipdb.set_trace()
       
    ## main loop
    lmbdas = [0]*nsub
    best_primal = +np.inf
    primals, duals, alphas, pdgaps = [], [], [], []
    for iter in range(niter):
        print 'iteration', iter
        dual = 0
        
        ## solve each subproblem (parallel)
        if is_parallel:
            data = []
            for k, subp in enumerate(subproblems_matrices):
                sub, Pk_bar, qk_bar, gtk_bar = subp
                qk_bar_ = qk_bar + lmbdas[k]
                info = {'k': k}
                data.append((info, nlabel, Pk_bar, qk_bar_, gtk_bar))
            #import ipdb; ipdb.set_trace()
            rdata = svm_worker.broadcast('duald',data)
            xks = []
            for info, xk, par_dual in rdata:
                xks.append(xk)
                dual += par_dual
            
            # data = [[] for s in range(MPISIZE)]
            # for k, subp in enumerate(subproblems_matrices):
                # sub, Pk_bar, qk_bar, gtk_bar = subp
                # qk_bar_ = qk_bar + lmbdas[k]
                # k_MPI = np.mod(k,MPISIZE)
                # info = {'k': k}
                # data[k_MPI].append((info, nlabel, Pk_bar, qk_bar_, gtk_bar))
            # data = Comm.scatter(data, root=0)            
            # rdata = Worker.solve_list_subproblems(data)
            # rdata = Comm.gather(rdata, root=0)
            # orders = []
            # xks_ = []
            # for rdata_ in rdata:
                # for info,xk,par_dual in rdata_:
                    # orders.append(info['k'])
                    # xks_.append(xk)
                    # dual += par_dual
            # xks = [xks_[i] for i in np.argsort(orders)]
            
        else:
            xks = []
            for k, subp in enumerate(subproblems_matrices):
                sub, Pk_bar, qk_bar, gtk_bar = subp
                qk_bar_ = qk_bar + lmbdas[k]
                xk = solver_gt(nlabel, Pk_bar, qk_bar_, gtk_bar, **kwargs)
                xks.append(xk)
                dual += float(0.5*xk.T*Pk_bar*xk + xk.T*qk_bar_)
        duals.append(dual)
        #print dual
        
        ## compute average
        x = np.mat(np.zeros((nvar,1)))
        for xk,sub in zip(xks,subproblems_matrices):
            if len(xk)!=len(sub[0]): 
                1/0
                #import pdb; pdb.set_trace()
            x[sub[0]] += xk.A/np.c_[N_bar[sub[0]]]

        ## test stop condition
        primal = float(0.5*x.T*P_bar*x + x.T*q_bar)
        primals.append(primal)
        if (primal - dual) < epsilon:
            stop = 0
            print 'stop condition reached at iteration {}'.format(iter)
            break

        if best_primal > primal:
            best_primal = primal
        pdgap = best_primal - dual
        print 'gap', pdgap
        pdgaps.append(pdgap)
        alpha = gamma*pdgap/float(x[N_bar>1].T*x[N_bar>1])
        alphas.append(alpha)
        
        ## update lambdas
        for k,sub in enumerate(subproblems_matrices):
            lmbdas[k] = lmbdas[k] + alpha*(xks[k] - x[sub[0]])
    
        # for k,sub in enumerate(subproblems_matrices):
            # seggt = np.argmax(gt_bar[sub[0]].reshape((nlabel,-1)), axis=0)
            # segx = np.argmax(xks[k].reshape((nlabel,-1)), axis=0)
            # if np.any(seggt!=segx):
                # import ipdb; ipdb.set_trace()
    
        # seggt = np.argmax(gt_bar.reshape((nlabel,-1)), axis=0)
        # segx = np.argmax(x.reshape((nlabel,-1)), axis=0)
        # if np.any(seggt!=segx):
            # import ipdb; ipdb.set_trace()
        stop = iter
        
    #if is_parallel:
    #    Comm.scatter([[] for i in range(MPISIZE)], root=0)
    
    info = {
        'primals': primals,
        'duals': duals,
        'alphas': alphas,
        'pdgap': pdgaps,
        'stop': stop,
        }
    return x, info

##------------------------------------------------------------------------------
def solver_gt(nlabel, P_bar, q_bar, gt_bar, **kwargs):
    P = P_bar
    q = q_bar
    
    gt_cols = gt_bar.reshape((nlabel,-1)).T
    
    ''' Gx >= h '''
    # ground truth constraints
    npixel = P.shape[0]/nlabel
    S = np.cumsum(gt_cols,axis=1)
    i_ = np.where(np.logical_not(gt_cols))
    rows = ((i_[1]-S[i_])*npixel + i_[0]).A.ravel()
    cols =  (i_[1]*npixel + i_[0]).A.ravel()
    
    G = sparse.coo_matrix(
        (-np.ones(len(rows)), (rows,cols)),
        shape=(npixel*(nlabel-1),npixel*nlabel),
        ).tocsr()
    
    G = G + sparse.bmat(
        [[sparse.spdiags(gt_cols[:,l].A.ravel(),0,npixel,npixel) for l in range(nlabel)] \
            for l2 in range(nlabel-1)])
    
    h = kwargs.pop('ground_truth_margin',1e-5)

    # positivity constraint
    G = sparse.bmat([[G], [sparse.eye(*P.shape)]])

    # sum-to-one constraint
    F = sparse.bmat([
        [sparse.bmat([[-sparse.eye(npixel,npixel) for i in range(nlabel-1)]])],
        [sparse.eye(npixel*(nlabel-1),npixel*(nlabel-1))],
        ])

    # solver with mosek
    objective = solver.ObjectiveAPI(P, q, G=G, h=h,F=F, **kwargs)
    constsolver = solver.ConstrainedSolver(objective)
    x = constsolver.solve(gt_bar) 
    
    return x    
    
##------------------------------------------------------------------------------
def compute_laplacian(
        image, 
        marked=None,
        beta=1.,
        ):
    
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
    data = weight_std(im,i,j,beta=beta)
        
    ## affinity matrix
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
        
    return Lu, B, D, border
    
##------------------------------------------------------------------------------
def weight_std(image, i, j, beta=1.0, offset=1e-8):
    ''' standard weight function 
    
        for touching pixel pair (i,j),
            wij = exp (- beta (image.flat[i] - image.flat[j])^2)
    '''
    im = np.asarray(image)
    wij = (1-offset)*np.exp(-beta * (im.flat[i] - im.flat[j])**2) + offset
    return wij
##------------------------------------------------------------------------------
        
        

        
if __name__=='__main__':

    kwargs = {
        'niter': 40,
        'gamma': 1e-2,
        'epsilon': 1e-2,
       }
       
    ## start workers
    if MPIRANK != 0:
        worker = Worker()
    else:
        import time
        ## load image
        # import os
        # if not os.path.isfile('test/im.npy'):
            # shape = (30,30)
            ##shape = (4,4)
            # im = np.random.random(shape)
            # nlabel = 3
            # gt = np.random.randint(0,nlabel,im.size).reshape(im.shape)
            # seeds = -np.ones(im.shape, dtype=int)
            # seeds.flat[0] = 0
            # seeds.flat[seeds.size/2] = 1
            # seeds.flat[-1] = 2
            # np.save('test/im.npy', im)
            # np.save('test/seeds.npy', seeds)
            # np.save('test/gt.npy', gt)
        # else:
            # im = np.load('test/im.npy')
            # seeds = np.load('test/seeds.npy')
            # gt = np.load('test/gt.npy')
        
        import io_analyze
        reslice = (slice(30,80), slice(30,80))
        im = io_analyze.load('test/thigh.hdr')[reslice]
        seeds = io_analyze.load('test/seeds.hdr')[reslice] - 1
        gt = io_analyze.load('test/gt.hdr')[reslice]
        
        im = im/np.std(im)
        
        labelset = np.unique(seeds[seeds>-1])
        nlabel = len(labelset)
        npixel = im.size
        
        ## make laplacian
        marked = np.where(seeds.ravel()>=0)[0]
        unknown = np.setdiff1d(np.arange(npixel), marked)
        print 'number of unknown pixels {}'.format(len(unknown))
        Lu, B, D, border = compute_laplacian(im, marked, beta=50)
        
        ## extend matrices 
        Lu_bar = sparse.kron(np.eye(nlabel),Lu)
        B_bar = sparse.kron(np.eye(nlabel),B)
        xm_bar = np.zeros((nlabel,border.size))
        for il,label in enumerate(labelset):
            xm_bar[il,:] = (seeds.flat==label)[border]
        xm_bar = np.mat(xm_bar.ravel()).T
        
        gt_bar = np.mat((gt.flat[unknown]==np.c_[labelset]).ravel()).T
        
        ## make qp
        P_bar = Lu_bar
        q_bar = - B_bar * xm_bar 
                                            
        ## run standard solver
        print 'compute standard method'
        start_time = time.time()
        x = solver_gt(nlabel, P_bar, q_bar, gt_bar)
        time_taken = time.time() - start_time
        print 'objective (standard method)', float(0.5 * x.T*P_bar*x + x.T*q_bar)
        print 'time spent: {}s'.format(time_taken)
        
        
        p = x.reshape((nlabel, -1))
        seg = np.zeros(im.shape, dtype=int)
        seg.flat[unknown] = labelset[np.argmax(p, axis=0)]
        
        ## run dual decomposition solver  
        # decomposes into subproblems
        print 'decompose problem'
        #subproblems, publicvars = decompose_brute_force(len(unknown),nlabel)   
        subproblems, publicvars = \
            decompose_with_image_connectivity(im.shape, nlabel, 
            marked=marked,
            size_sub=10)
            
        imsub = np.zeros(im.shape, dtype=int)
        for i,sub in enumerate(subproblems):
            imsub.flat[unknown[sub[:len(sub)/nlabel]]] = i
        io_analyze.save('imsub.hdr', imsub.astype(np.int32))
            
        # dd solve
        print 'compute dd method'
        start_time = time.time()
        x_d, info = dd_solver_gt(
            nlabel, 
            Lu, q_bar, gt_bar, 
            subproblems,
            **kwargs)
        time_taken = time.time() - start_time    
        print 'objective (dual d. method)', float(0.5 * x_d.T*P_bar*x_d + x_d.T*q_bar)
        print 'time spent: {}s'.format(time_taken)
        print 'number of iterations: {}'.format(info['stop'])
        
        p_d = x_d.reshape((nlabel, -1))
        seg_d = np.zeros(im.shape, dtype=int)
        seg_d.flat[unknown] = labelset[np.argmax(p_d, axis=0)]
        
        print 'max diff', np.max(np.abs(p - p_d))
        print 'label differences ?', np.where(seg.ravel()!=seg_d.ravel())[0]
        
        from matplotlib import pyplot
        fig = pyplot.figure()
        pyplot.plot(info['primals'], 'o-g', figure=fig, label='primal')
        pyplot.plot(info['duals'], 'o-b', figure=fig, label='dual')
        pyplot.plot(info['pdgap'], 'o-y', figure=fig, label='gap')
        pyplot.legend()
        
        #fig2 = pyplot.figure()
        #pyplot.plot(info['alphas'], 'o-b', figure=fig2)
        
        pyplot.show()
    
    

    
     
    
    
    
