import numpy as np
from scipy import sparse

class ObjectiveAPI(object):
    def __init__(self,L,wprior,y0,z,**kwargs):
        
        self.P = L + wprior*sparse.eye(*L.shape)
        self.q = -wprior*y0 + z
        self.c = 0.5*wprior*np.dot(y0,y0)
        
        ## compute diagonal P
        ijP   = P.nonzero()
        diagP = np.eq(*ijP)
        self.diagP = sparse.coo_matrix(
            (P.data[diagP], (ijP[0][diagP], ijP[1][diagP])),
            shape=self.P.shape,
            )
        
        # nlabel = 
        # npixel = 
        
        # self.F = F
        ## probability constraints (null space)
        # F = sparse.bmat([
            # [-sparse.eye(npixel,npixel) for i in range(nlabel-1)],
            # [sparse.eye(npixel*(nlabel-1),npixel*(nlabel-1))],
            # ])


    def __call__(self,x):
        return 0.5 * x.T * self.P * x + self.q.T * x + self.c
            
    def gradient(self,x):
        'Px + q'
        return self.P * x + self.q
 
    def hessian(self,x):
        'P'
        return self.P
        
    def hessian_diag(self,x):
        return self.diagP
        

class ModifiedGradientDescent(object):
    def __init__(self, objective, gradient, hessian_diag, **kargs):
        self.objective = objective
        self.gradient = gradient
        self.hessian_diag = hessian_diag
        
        ## line search parameters
        self.a = kargs.pop('a', 0.4)
        self.b = kargs.pop('b', 0.9)
        
        self.epsilon = kwargs.pop('epsilon', 1e-6)
        self.maxiter = kwargs.pop('maxiter', 100)
        
    def solve(self, x0):
        epsilon = self.epsilon
        x = x0
        nvar = len(x0)
        
        for iter in range(self.maxiter):
            
            ## compute gradient and Hessian
            gradx = self.gradient(x)
            invH = sparse.spdiags(1.0/self.hessian_diag(x), 0, nvar, nvar)
            
            ## Newton's step and increment
            x_nt  = -invH * gradx
            lmbda2 = - np.dot(gradx, x_nt)

            ## stopping conditions
            if 0.5*lmbda2 <= epsilon:
                return x

            ## line search 
            step = self.line_search(x,x_nt,gradx)

            ## update
            x = x + step*x_nt
            
        else:
            raise Exception('Did not converge in {} iterations'\
                .format(self.maxiter))
        
        
    def linesearch(self,x,dx,gradx):
        t = 1
        objective = self.objective
        while objective(x + t*dx) > objective(x) + self.a * t * gradx.T * dx:
            t = b * t
        return t
        
        

        
        
class ConstrainedSolver(object):
    def __init__(self, objective, F, **kargs):
        self.objective = objective
        self.epsilon = kwargs.pop('epsilon',1e-6)
        
    def solve(self,y,t):
    
        objective = self.objective
        objective_t = lambda u: \
            t * objective(F*u + y) - np.sum(np.log(F*u + y))
    
        gradient = self.objective.gradient
        gradient_t = lambda u: \
            self.F.T * (t * gradient(F*u + y) - 1.0/(F*u + yhat))
        
        hessian = self.objective.hessian
        hessian_t = lambda: u: \
            self.F.T * (t * hessian(F*u + y) + \
                        self._diag(1.0/(F*u + y)**2)) * self.F
        
        
        solver = ModifiedGradientDescent(
            objective_t, gradient_t, hessian_t, epsilon=self.epsilon)
        
    def _diag(self,x):
        n = x.size
        return sparse.spdiags(np.asarray(x).ravel(),0,n,n)
        
        
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
            y = self.solver(y,t)

            ## stopping conditions
            # Note: the number of constraints in the original algorithm
            # is absorbed inside epsilon
            if 1./t < epsilon:
                break

            ## update t
            t = mu * t

        return y

