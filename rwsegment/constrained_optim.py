import numpy as np
from scipy import sparse

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
        self.b = kwargs.pop('b', 0.9)
        
        self.epsilon = kwargs.pop('epsilon', 1e-6)
        self.maxiter = kwargs.pop('maxiter', 100)
        
    def solve(self, x0):
        epsilon = self.epsilon
        x = x0
        nvar = len(x0)
        
        for iter in range(self.maxiter):
            
            ## compute gradient and Hessian
            gradx = self.gradient(x)
            # invH = sparse.spdiags(1.0/self.hessian_diag(x), 0, nvar, nvar)
            invH = sparse.spdiags(
                1.0 / self.extract_diag(self.hessian(x)),0,nvar,nvar)
            
            ## Newton's step and increment
            x_nt  = -invH * gradx
            lmbda2 = - np.dot(gradx.T, x_nt)

            ## stopping conditions
            if 0.5*lmbda2 <= epsilon:
                return x

            ## line search 
            step = self.line_search(x,x_nt,gradx)
            
            logger.debug(
                'MGD: iteration={}, step size={}, lambda2={}'\
                .format(iter, step, lmbda2),
                )
            
            ## update
            x = x + step*x_nt
            
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
        
        
    def line_search(self,x,dx,gradx):
        t = 1
        objective = self.objective
        while objective(x + t*dx) > objective(x) + self.a * t * gradx.T * dx:
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
        objective_t = lambda u: \
            t * objective(F*u + y) - np.sum(np.log(F*u + y))
    
        gradient = self.objective.gradient
        gradient_t = lambda u: \
            F.T * (t * gradient(F*u + y) + \
                        self.barrier_gradient(F*u + y))
        
        hessian = self.objective.hessian
        hessian_t = lambda u: \
            F.T * (t * hessian(F*u + y) + \
                        self.barrier_hessian(F*u + y)) * self.F
        
        logger.debug('calling MGD solver')
        solver = ModifiedGradientDescent(
            objective_t, gradient_t, hessian_t, epsilon=self.epsilon)
            
        nvar = F.shape[1]
        u0 = np.matrix(np.zeros((nvar,1)))
        
        return F*solver.solve(u0) + y
        
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
            ## solve with currant t
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