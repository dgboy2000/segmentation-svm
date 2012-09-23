import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

class ObjectiveAPI(object):
    def __init__(self,P,q,G=1,h=0,F=1,**kwargs):
    
        self.P = P
        self.q = q
        self.iconst = (G,h)
        self.eqconst = F
        
        ## constant term
        self.c = kwargs.pop('c',0)

    def __call__(self,x):
        return 0.5 * x.T * self.P * x + self.q.T * x + self.c
            
    def gradient(self,x):
        'Px + q'
        return self.P * x + self.q
 
    def hessian(self,x):
        'P'
        return self.P
        

class ConstrainedSolver(object):
    def __init__(self, objective, **kwargs):
        '''
            objective: objective function
                objective() evaluates the objective
                objective.gradient() evalutes its gradient
                objective.hessian() evaluates its hessian
                objective.iconst is (G,h) 
                    for enforcing linear constraints: Gx >= h
                    G and h can be scalars if the same constraints apply 
                    to all variables
                objective.eqconst is F 
                    where A enforces linear constraints: Ax = b
                    and AF = 0
                    the initial guess in solve(x0) must satisfy A x0 = b
                    set F to None if no equality constraints
            
                '''
                
        self.objective = objective
        
        self.epsilon   = kwargs.pop('epsilon',1e-3)
        self.t0 = kwargs.pop('t0',1)
        self.mu = kwargs.pop('mu',20)
        self.maxiter = kwargs.pop('maxiter',100)
        
        ## inequality constraints
        G, h = objective.iconst
        if not hasattr(G,'T'):
            ## scalar G
            self.G  = G
            self.GT = G
            self.h = h
        else:
            ## matrix G
            self.G  = G
            self.GT = G.T
            self.h = h
        
        ## equality constraints
        self.nvar = np.inf
        F = objective.eqconst
        if F is None or not hasattr(F,'T'):
            ## no equality constraints
            self.F  = 1.0
            self.FT = 1.0
        else:
            ## matrix F
            self.F  = F
            self.FT = F.T
            self.nvar = np.minimum(F.shape[1], self.nvar)
        
    def solve(self, x0, **kwargs):
        ''' (parameters for the Newton solver in kwargs)
            initial guess x0 has to satisfy the equality constraints !
            Ax0 = b
        '''
        
        epsilon = self.epsilon
        mu      = self.mu
        t0      = self.t0
        # newt_epsilon = kwargs.pop('epsilon', None)

        x = x0
        t = t0

        G,h = self.G, self.h
        self.nconst = (G*x0).shape[0]
        
        nvar = min(self.nvar, x0.size)
        
        ## non frozen indices
        self.alive = np.arange(x0.size)
        
        ## test inequality constraints
        inconst = np.min(G*x0 - h)
        try:
            assert inconst > 0
        except:
            import sys
            logger.error('Inequality constraint not satisfied, by {:.3}'.format(inconst))
            import ipdb; ipdb.set_trace()
            sys.exit(1)
        
        for iter in range(self.maxiter):
            ## solve with currant t
            logger.debug(
                'calling constrained solver with t={}'.format(t))
                
            modf_objective = lambda u: self.barrier_objective_eqc(u,x,t)
            modf_gradient  = lambda u: self.barrier_gradient_eqc(u,x,t)
            modf_hessian   = lambda u: self.barrier_hessian_eqc(u,x,t)
               
            ## Newton's method
            # increase epsilon with t
            # if newt_epsilon is None:
                #eps = np.maximum(1e-3,t*1e-6)
                # eps = 1e-6
            # else:
                # eps = newt_epsilon
            # logger.debug(
                # 'calling Newton solver with tol={:.2}'.format(newt_epsilon))
            
            # instanciate solver
            solver = NewtonMethod(
                modf_objective, 
                modf_gradient, 
                modf_hessian,
                # epsilon=newt_epsilon,
                **kwargs)
                
            # solve with Newton' method
            u0 = np.asmatrix(np.zeros(nvar)).T
            u = solver.solve(u0)
            
            F = self.F
            x = F*u + x
            
            ## stopping conditions
            if (self.nconst/float(t)) < epsilon:
                return x

            ## update t
            t = mu * t

        return y
        
    def barrier_objective_eqc(self,u,x0,t):
        objective = self.objective
        F   = self.F
        G,h = self.G, self.h
        x = F*u + x0
        cond = G*x - h
        if np.min(cond)<= 0:
            ## in case of negative values, return infinity
            return np.infty
        #logger.debug('min cond ={}'.format(np.min(cond)))
        return objective(x) - 1./t * np.sum(np.log(cond))
    
    def barrier_gradient_eqc(self,u,x0,t):
        gradient = self.objective.gradient
        F   = self.F
        G,GT,h = self.G, self.GT, self.h
        FT = self.FT
        x = F*u + x0
        cond = G*x - h
        #return FT * (t * gradient(x) + GT * self.barrier_gradient(cond))
        return FT * ( gradient(x) + 1.0/t * GT * self.barrier_gradient(cond))
        
    def barrier_gradient(self,cond):
        return np.matrix(-1.0/np.asarray(cond).ravel()).T
        
    
    def barrier_hessian_eqc(self,u,x0,t):
        hessian = self.objective.hessian
        F   = self.F
        G,GT,h = self.G, self.GT, self.h
        FT = self.FT
        x = F*u + x0
        cond = G*x - h
        #return FT * (t * hessian(x) + \
        #              GT * self.barrier_hessian(cond) * G) * F
        return FT * (hessian(x) + \
                      1./t * GT * self.barrier_hessian(cond) * G) * F
 
    def barrier_hessian(self,cond):
        n = cond.size
        return sparse.spdiags(1.0/np.asarray(cond).ravel()**2,0,n,n)
    
        
class NewtonMethod(object):
    def __init__(self, objective, gradient, hessian, **kwargs):
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        
        ## line search parameters
        self.a = kwargs.pop('a', 0.4)
        self.b = kwargs.pop('b', 0.8)
        
        self.epsilon = kwargs.pop('epsilon', 1e-6)
        self.maxiter = int(kwargs.pop('maxiter', 100))
        
        self.use_diagonal_hessian = kwargs.pop('use_diagonal_hessian',False)
        
    def solve(self, u0):
        epsilon = self.epsilon
        u = u0
        u_nt = np.mat(np.zeros((len(u),1)))
        lmbda2 = 1e10
        nvar = len(u0)
        
        logger.debug(
                'starting Newton solver with tol={:.2}'.format(epsilon))
                
        for iter in range(self.maxiter):
            ## compute gradient and Hessian
            logger.debug('Newton: compute direction')
            gradu = self.gradient(u)
            Hu = self.hessian(u)
 
            ## Newton step
            if self.use_diagonal_hessian:
                ## Modified gradient descent
                invH = sparse.spdiags(1.0 / self.extract_diag(Hu),0,nvar,nvar)
                info = 0
                u_nt  = -invH * gradu
            else:
                ## standard Newton's Method
                import gc
                ## starting scheme
                l0 = float(np.dot(gradu.T, u_nt))
                if l0 < -1e-8:
                    u_nt0 = float(-1.0/(u_nt.T*Hu*u_nt) * l0) * u_nt
                else: 
                    u_nt0 = np.mat(np.zeros((len(u),1)))
                tol = np.minimum(1e-1, np.sqrt(np.sqrt(float(np.dot(gradu.T, gradu)))))
                #M = sparse.spdiags(self.extract_diag(Hu), 0, *Hu.shape)
                #Hu = (Hu + sparse.spdiags(np.sqrt(np.sum(gradu.A**2)), 0, *Hu.shape)).tocsr() ## regularized newton method
                u_nt,info = splinalg.cg(Hu, -gradu, x0=u_nt0, tol=tol, maxiter=1000)
                u_nt = np.asmatrix(u_nt).T
                if info > 0:
                    invH = sparse.spdiags(1.0 / self.extract_diag(Hu),0,nvar,nvar)
                    u_nt = -invH * gradu
                gc.collect()
            
            
            ## Newton increment
            lmbda2 = - np.dot(gradu.T, u_nt)
            #if lmbda2_ > lmbda2: self.b  = np.sqrt(self.b)
            #else: lmbda2 = lmbda2_
            #if lmbda2_ > lmbda2: self.a  = 0.9 * self.a
            #else: lmbda2 = lmbda2_


            ## stopping conditions
            if (0.5*lmbda2 <= epsilon): 
                logger.debug(
                    'Newt: return, lambda2={:.02}'.format(float(lmbda2)))
                #import ipdb; ipdb.set_trace()
                return u

            ## line search 
            logger.debug('Newton: (cg info = {}), a={:.3}, b={:.3}, line search'.format(info, float(self.a), float(self.b)))
            step = self.line_search(u,u_nt,gradu)
            
            logger.debug(
                'Newt: iter={}, step size={:.02}, lambda2={:.02}, obj={:.02}'\
                .format(iter,float(step),float(lmbda2),float(self.objective(u))),
                )
            
            # import ipdb; ipdb.set_trace()
            
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
        b = float(self.b)
        while objective(u + t*du) > objective(u) + self.a * t * gradu.T * du:
            t = b * t
        return t
        
        
import utils_logging
logger = utils_logging.get_logger(
    'solver_qp_constr',utils_logging.DEBUG)
