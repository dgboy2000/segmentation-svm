import sys
import numpy as np
from scipy import sparse

import mosek

class ObjectiveAPI(object):
    def __init__(self,P,q,G=1,h=0,F=1,**kwargs):
    
        self.P = P
        self.q = q
        self.iconst = G,h
        self.G, self.h = G,h
        self.eqconst = F
        self.F = F

        self.nvar = q.size        
        self.nconst = 1
        if hasattr(G, 'shape'):
            self.nconst = G.shape[0]
        if hasattr(F, 'shape'):
            self.nvar = F.shape[1]

        ## constant term
        self.c = kwargs.pop('c',0)

    def get_Qc(self):
        if hasattr(self.F, 'shape'): 
            FT = self.F.T
        else:
            FT = self.F
        Q =  sparse.tril(FT * self.P * self.F)
        c = FT*self.q
        return Q,c

    def get_Ab(self):
        G,h = self.iconst
        A = G*self.F
        b = np.multiply(h, np.mat(np.ones((G.shape[0],1))))
        return A,b

    def get_x(self, u, x0):
        return self.F*u + x0

    def get_const(self, x):
        return self.G*x - self.h

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
        self.maxiter = kwargs.pop('maxiter',100)

    def solve(self, x0, **kwargs):

        A,b = self.objective.get_Ab()
        Q,c = self.objective.get_Qc()


        #mosek.iparam.log = 1
        
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
        task.set_Stream (mosek.streamtype.log, streamprinter)
        
        NUMVAR = self.objective.nvar
        NUMCON = self.objective.nconst
        NUMANZ = A.nnz
       
        task.putmaxnumvar(NUMVAR) 
        task.putmaxnumcon(NUMCON) 
        task.putmaxnumanz(NUMANZ)
        task.append(mosek.accmode.con,NUMCON)
        task.append(mosek.accmode.var,NUMVAR)

        # Set up and input bounds and linear coefficients
        #bkx = np.asarray([mosek.boundkey.up for i in range(NUMVAR)])
        #blx = np.[ 0.0 for i in range(NUMVAR)]
        #bux = [ +inf for i in range(s)]

        # Set the bounds on variable j 
        # blx[j] <= x_j <= bux[j] 
        #task.putboundlist(mosek.accmode.var,range(NUMVAR),bkx,blx,bux)
 
        # quadratic term    
        qsubi, qsubj = Q.nonzero()
        qval = Q.data 
  
        # Set the linear term c_j in the objective. 
        task.putclist(range(NUMVAR),c.A.ravel())
        
        # Set up and input quadratic objective
        task.putqobj(qsubi,qsubj,qval)

        ## const term
        task.putcfix(0.0)
        
        ## set constraints
        asubi, asubj = A.nonzero()
        aval = A.data
        for iconst in np.unique(asubi):
            inds = np.where(asubi==iconst)[0]
            ivar_i = asubj[inds]
            data_i = aval[inds]
            
            task.putavec (
                mosek.accmode.con,
                iconst,
                ivar_i,
                data_i,
                ) 

            ## set bounds on constraints
            task.putbound(
                mosek.accmode.con,
                iconst,
                mosek.boundkey.up,
                -inf,
                b.A[iconst],
                ) 

        # Input the objective sense (minimize/maximize) 
        task.putobjsense(mosek.objsense.minimize)
        
        import ipdb; ipdb.set_trace()
        # Optimize 
        task.optimize() 
        
        # Print a summary containing information 
        # about the solution for debugging purposes 
        #task.solutionsummary(mosek.streamtype.msg)
        
        prosta = [] 
        solsta = [] 
        [prosta,solsta] = task.getsolutionstatus(mosek.soltype.itr)
        
        # Output a solution 
        xx = [0 for i in range(NUMVAR)] 
        task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0,NUMVAR, xx)
      
        x = np.mat(xx).T 
        return x

