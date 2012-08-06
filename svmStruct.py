 

class StructSVM(object):
    def __init__(
            self, 
            training_set, 
            loss_function, 
            psi, 
            most_violated_constraint,
            **kwargs):
        ''' 
            S = [(x1,y1),..., (xn,yn)]
            
            args:
                loss    : (scalar) = loss(y,y_)
                psi     : (vector) = psi(x,y)
                most_violated_constraint : y_ = most_violated_constraint(w,x,y)
        '''
        self.S = training_set
        self.C = kwargs.pop('C',1.)
        self.w_loss  = kwargs.pop('w_loss',1.)
        self.epsilon = kwargs.pop('epsilon',1e-5)
        
        self.loss = loss_function
        self.psi  = psi
        self.mvc  = most_violated_constraint
        
        
    
    def _current_solution(self, W):
        ''' quadratic programming 
        min ztPz + 2*ztq 
            st. Az <= b
        
        Here:
        min 1/2wtw + Cxi
            st. for all (y1_,...yn_) in W
                1/n wt sum(k){psi(xk,yk) - psi(xk,yk_)} 
                    >= 1/n sum(k) {loss(yk,yk_) - xi
        '''
        import mosek
        import sys
        def streamprinter(text): 
            sys.stdout.write(text) 
            sys.stdout.flush()
            
        # Open MOSEK and create an environment and task 
        # Make a MOSEK environment 
        env = mosek.Env () 
        
        # Attach a printer to the environment 
        env.set_Stream (mosek.streamtype.log, streamprinter) 
        
        # Create a task 
        task = env.Task() 
        task.set_Stream (mosek.streamtype.log, streamprinter)
        
        # Set up and input bounds and linear coefficients
        bkx = [mosek.boundkey.fr for i in range(self.wsize)] + \
              [mosek.boundkey.lo]
        blx = [ -inf for i in range(self.wsize)] + [0.0]
        bux = [ +inf for i in range(self.wsize)] + [+inf]
        
        c = [0 for i in range(len(W))] + [self.C]
        
        qsubi = range(self.wsize)
        qsubj = qsubi
        qval = [1.0]*self.wsize
        
        NUMVAR = len(bkx) 
        NUMCON = len(W)
        NUMANZ = 3
        
        task.putmaxnumvar(NUMVAR) 
        task.putmaxnumcon(NUMCON) 
        task.putmaxnumanz(NUMANZ)
        
        task.append(mosek.accmode.con,NUMCON)
        task.append(mosek.accmode.var,NUMVAR)
        task.putcfix(0.0)
        
        ## psi(x,z)
        Ssize = len(self.S)
        avg_phi_gt = [0 for i in range(self.wsize)]
        for s in self.S:
            for i_p,p in enumerate(self.psi(*s)): 
                avg_phi_gt[i_p] += p / float(Ssize)
        
        ## set the constraints
        for j in range(NUMCON): 
         
            ## psi(x,y_)
            avg_phi = [0 for i in range(self.wsize)]

            avg_loss = 0
            for i_y,y_ in enumerate(W[j]):
                # average loss
                avg_loss += self.loss(self.S[i_y][1], y_) / float(Ssize)
                
                # average psi
                for i_p,p in enumerate(self.psi(self.S[i_y][0],y_)): 
                    avg_phi[i_p] += p / float(Ssize)
            
            ## psi(x,y_) - psi(x,z)
            aval = \
                [avg_phi[i] - avg_phi_gt[i] for i in range(self.wsize)] + \
                [1.0]
         
            ## Input row j of A 
            task.putavec(
                mosek.accmode.con, # Input rows of A. 
                j,           # Variable (column) index. 
                range(NVAR), # Column index of non-zeros in column j. 
                aval,        # Non-zero Values of row j. 
                )
        
            ## set bounds on constraints
            task.putbound(
                mosek.accmode.con,
                j,
                mosek.boundkey.lo,
                avg_loss,
                +inf,
                )
        
        # Set the bounds on variable j 
        # blx[j] <= x_j <= bux[j] 
        task.putboundlist(mosek.accmode.var,range(NVAR),bkx,blx,bux)
        
        # Set the linear term c_j in the objective. 
        task.putclist(range(NVAR),c)
        
        # Set up and input quadratic objective
        task.putqobj(qsubi,qsubj,qval)
        
        # Input the objective sense (minimize/maximize) 
        task.putobjsense(mosek.objsense.minimize)
        
        # Optimize 
        task.optimize() 
        
        # Print a summary containing information 
        # about the solution for debugging purposes 
        task.solutionsummary(mosek.streamtype.msg)
        
        prosta = [] 
        solsta = [] 
        [prosta,solsta] = task.getsolutionstatus(mosek.soltype.itr)
        
        # Output a solution 
        xx = np.zeros(NUMVAR, float) 
        task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0,NUMVAR, xx)
        
        w,xi = xx[:self.wsize], xx[-1]
        
        return w,xi
        
    def _stop_condition(self,w,xi):
        pass
        
    def train(self):
        ''' optimize with algorithm:
        "Cutting-plane training of structural SVMs"
        Machine Learning 2009
        Joachims, Thorsten
        Finley, Thomas 
        Yu, Chun-Nam John
        '''
        
        ## test set for qp
        W = [] 
        
        ## initialize w
        self.wsize = len(self.psi(*self.S[0]))
        w = 0
        
        niter = 0
        while 1:
            ## compute current solution (qp + constraints)
            w,xi = self._current_solution(W)
        
            ## find most violated constraint
            ys = []
            for s in self.S:
                y_ = self.mvc(w, *s)
                ys.append(y_)
                
            ## add to test set
            W.append(ys)
            
            ## stop condition
            if self._stop_condition(w,xi): break
            else: niter+= 1
        
        ## return values
        info = {
            'number of iterations': niter, 
            'number of contraints': len(W),
            }
        return w, xi, info
        
        
if __name__=='__main__':
    
    ## test svm struct
    
    pass
    