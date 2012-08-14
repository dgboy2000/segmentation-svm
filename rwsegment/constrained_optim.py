import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

class ObjectiveAPI(object):
    def __init__(self,P,q,**kwargs):
    
        self.P = P
        self.q = q

    def __call__(self,x):
        return 0.5 * x.T * self.P * x + self.q.T * x #+ self.c
            
    def gradient(self,x):
        'Px + q'
        return self.P * x + self.q
 
    def hessian(self,x):
        'P'
        return self.P
        

class ModifiedGradientDescent(object):
    def __init__(self, objective, gradient, hessian, **kwargs):
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        
        ## line search parameters
        self.a = kwargs.pop('a', 0.4)
        self.b = kwargs.pop('b', 0.8)
        
        self.epsilon = kwargs.pop('epsilon', 1e-6)
        self.maxiter = kwargs.pop('maxiter', 100)
        
        self.use_diagonal_hessian = kwargs.pop('use_diagonal_hessian',False)
        
    def solve(self, u0):
        epsilon = self.epsilon
        u = u0
        nvar = len(u0)
        
        for iter in range(self.maxiter):
            
            ## compute gradient and Hessian
            gradu = self.gradient(u)
            Hu = self.hessian(u)
            
            ## Newton's step and increment
            import gc
            u_nt,info = splinalg.cg(Hu,-gradu, tol=1e-3, maxiter=1000)
            # u_nt,info = splinalg.bicg(Hu,-gradu, tol=1e-3, maxiter=1000)
            # u_nt,info = splinalg.gmres(Hu,-gradu, tol=1e-3, maxiter=1000)
            # u_nt,info = splinalg.lgmres(Hu,-gradu, tol=1e-3, maxiter=1000)#!
            # u_nt,info = splinalg.cgs(Hu,-gradu, tol=1e-3, maxiter=1000)
            # u_nt,info = splinalg.minres(Hu,-gradu, tol=1e-3, maxiter=1000)
            # u_nt,info = splinalg.qmr(Hu,-gradu, tol=1e-3, maxiter=1000)
            # u_nt,info = splinalg.spsolve(Hu,-gradu)
            
            u_nt = np.asmatrix(u_nt).T
            gc.collect()
            
            ## Modified Gradient Descent
            # invH = sparse.spdiags(
                # 1.0 / self.extract_diag(Hu),0,nvar,nvar)
            # u_nt  = -invH * gradu
            
            lmbda2 = - np.dot(gradu.T, u_nt)
            
            ## stopping conditions
            if 0.5*lmbda2 <= epsilon:
                logger.debug(
                    'MGD: return, lambda2={:.02}'.format(float(lmbda2)))
                return u
            
            #if iter==0:
            #    logger.debug('iter=0, normgradu={}, normHu={}'\
            #        .format(np.dot(gradu.T,gradu), np.sum(invH.data**2)))
            
            ## line search 
            step = self.line_search(u,u_nt,gradu)
            
            logger.debug(
                'MGD: iteration={}, step size={:.02}, lambda2={:.02},obj={:.02}'\
                .format(iter, float(step), float(lmbda2),float(self.objective(u))),
                )
            
            ## update
            u = u + step*u_nt
            
        else:
            raise Exception('Did not converge in {} iterations'\
                .format(self.maxiter))
        
    def extract_diag(self,H):
        ## extract diagonal from H
        ijH     = H.nonzero()
        ondiagH = np.equal(*ijH)
        diagH   = np.zeros(H.shape[0])
        diagH[ijH[0][ondiagH]] = H.data[ondiagH]
        return diagH
        
        
    def line_search(self,u,du,gradu):
        t = 1.0
        objective = self.objective
        while objective(u + t*du) > objective(u) + self.a * t * gradu.T * du:
            t = self.b * t
        return t
        
        
        
class ConstrainedSolver(object):
    def __init__(self, objective, F, **kwargs):
        self.objective = objective
        self.epsilon = kwargs.pop('epsilon',1e-6)
        self.F = F
        
    def solve(self,y,t):
    
        F = self.F
        objective = self.objective
        objective_t = lambda u: self._objective_t(u,y,t)
            # t * objective(F*u + y) - np.sum(np.log(F*u + y))
    
        gradient = self.objective.gradient
        gradient_t = lambda u: \
            F.T * (t * gradient(F*u + y) + \
                        self.barrier_gradient(F*u + y))
        
        hessian = self.objective.hessian
        hessian_t = lambda u: \
            F.T * (t * hessian(F*u + y) + \
                        self.barrier_hessian(F*u + y)) * self.F
        
        epsilon = np.maximum(1e-3,t*1e-6)
        logger.debug('calling MGD solver with tol={:.3}'.format(epsilon))
        solver = ModifiedGradientDescent(
            objective_t, gradient_t, hessian_t, epsilon=epsilon)
            
        nvar = F.shape[1]
        u0 = np.matrix(np.zeros((nvar,1)))
        
        return F*solver.solve(u0) + y
        
    def _objective_t(self,u,y,t):
        F = self.F
        if np.min(F*u + y)<= 0:
            ## in case of negative values, return infinity
            return np.infty
            # import ipdb; ipdb.set_trace()
        return t * self.objective(F*u + y) - np.sum(np.log(F*u + y))
        
    def barrier_hessian(self,x):
        n = x.size
        return sparse.spdiags(1.0/np.asarray(x).ravel()**2,0,n,n)
        
    def barrier_gradient(self,x):
        return np.matrix(-1.0/np.asarray(x).ravel()).T
        
        
class LogBarrierSolver(object):
    def __init__(self, constrained_solver, t0, mu, epsilon):
        self.solver = constrained_solver
        self.t0 = t0
        self.mu = mu
        # self.y0 = y0
        self.epsilon = epsilon
        
        
    def solve(self, y0):
        epsilon = self.epsilon
        mu      = self.mu
        t0      = self.t0

        y = y0
        t = t0

        while 1:
            ## solve with current t
            logger.debug(
                'calling constrained solver with t={}'.format(t))
            y = self.solver.solve(y,t)

            ## stopping conditions
            # Note: the number of constraints in the original algorithm
            # is absorbed inside epsilon
            if 1./t < epsilon:
                break

            ## update t
            t = mu * t

        return y


import rwlogging
logger = rwlogging.get_logger('coptimlogger',rwlogging.DEBUG)