import sys
import mosek
import utils_logging as logging
logger = logging.get_logger('svm_solver', logging.INFO)

class SVMSolver(object):
    
    def __init__(self, C, **kwargs):
        ''' solve quadratic program
        min ztPz + 2*ztq 
            st. Az <= b
        
        Here:
        min 1/2wtw + Cxi
            st. for all (y1_,...yn_) in W
                1/n wt sum(k){psi(xk,yk) - psi(xk,yk_)} 
                    >= 1/n sum(k) {loss(yk,yk_) - xi
                    
        Our data structure:
        W = [{'psis':[],'losses':[]}, {'psis':[],'losses':[]}]

        '''   
        self.C = C
        self.use_mosek = kwargs.pop('use_mosek', True)
        self.Cprime = kwargs.pop('Cprime', 0)

    def objective(self, w, xi, wref=None):
        obj = 0.5 * sum([val**2 for val in w]) + self.C * xi
        if wref is not None:
            obj += self.Cprime * sum([(w1-w2) for w1,w2 in wip(w,wref)])
        return obj

    def solve(self, W, gtpsis, **kwargs):
        NUMW = len(gtpsis[0])
        if len(W)==0:
            ## if no constraints return provided w, or w=0.
            w = kwargs.pop('w0', None)
            if w is None:
                w = [0.0 for i in range(NUMW)]
            xi = 0.0
            return w, xi

        scale_only = kwargs.pop('scale_only', False)
        if self.use_mosek:
            if scale_only:
                w0 = kwargs.pop('w0', [1.0 for i in range(NUMW)])
                return self.solve_mosek_scale(W, gtpsis, w0, **kwargs)
            else:
                return self.solve_mosek(W, gtpsis, **kwargs)
        else:
            return self.solve_no_mosek(W, gtpsis, **kwargs)

    def solve_mosek(self, W, gtpsis, **kwargs):
        wref = kwargs.pop('wref',None)

        def streamprinter(text): 
            sys.stdout.write(text) 
            sys.stdout.flush()
            dir  = logging.LOG_OUTPUT_DIR
            rank = logging.RANK
            if dir is not None:
                f = open('{}/output{}.log'.format(dir, rank),'a') 
                f.write(text)
                f.close()
        
        inf = 0
        
        # Open MOSEK and create an environment and task 
        # Make a MOSEK environment 
        env = mosek.Env () 
        
        # Attach a printer to the environment 
        env.set_Stream (mosek.streamtype.log, streamprinter) 
        
        # Create a task 
        task = env.Task() 
        task.set_Stream (mosek.streamtype.log, streamprinter)
 
        # problem dimensions
        NUMX   = len(gtpsis)
        NUMW   = len(gtpsis[0])
        NUMVAR = NUMW + 1 # +1 for xi
        NUMCON = len(W)
        NUMANZ = NUMCON*NUMVAR
        streamprinter('NUMVAR={}, NUMCON={}\n'.format(NUMVAR, NUMCON))
          
        # set const
        task.putmaxnumvar(NUMVAR) 
        task.putmaxnumcon(NUMCON) 
        task.putmaxnumanz(NUMANZ)
       
        task.append(mosek.accmode.con,NUMCON)
        task.append(mosek.accmode.var,NUMVAR)       
        bkx = [mosek.boundkey.lo for i in range(NUMW)] + \
              [mosek.boundkey.lo]
        blx = [ 0.0 for i in range(NUMW)]  + [0.0]
        bux = [ +inf for i in range(NUMW)] + [+inf]
        
        # linear term in objective
        c = [0 for i in range(NUMW)] + [self.C]

        # quadratic term in objective
        qsubi = range(NUMVAR)
        qsubj = qsubi
        qval = [1. for i in range(NUMW)] + [0]
        
        ## constant term
        cfix = 0

        if wref is not None:
            c = [val1 - self.Cprime*val2 for val1,val2 in zip(c, wref)]
            qval = [val + self.Cprime for val in qval]
            cfix += sum([self.Cprime*val**2 for val in wref])

        ## constant term
        task.putcfix(cfix)
        
        # Set the bounds on variable j 
        # blx[j] <= x_j <= bux[j] 
        task.putboundlist(mosek.accmode.var,range(NUMVAR),bkx,blx,bux)
        
        # Set the linear term c_j in the objective. 
        task.putclist(range(NUMVAR),c)
        
        # Set up and input quadratic objective
        task.putqobj(qsubi,qsubj,qval)

        ## psi(x,z)
        avg_psi_gt = [0 for i in range(NUMW)]
        for gtpsi in gtpsis:
            for i, p in enumerate(gtpsi):
                avg_psi_gt[i] += 1.0/ float(NUMX) * p

        ## set the constraints
        for j in range(NUMCON): 
            # average psi
            avg_psi = [0 for i in range(NUMW)]
            for psi in W[j]['psis']:
                for i,p in enumerate(psi):
                    avg_psi[i] += 1.0/float(NUMX) * p
                    
            # average loss
            avg_loss = 0
            for loss in W[j]['losses']:
                avg_loss += 1.0/float(NUMX) * loss
            
            ## psi(x,y_) - psi(x,z)
            aval = \
                [avg_psi[i] - avg_psi_gt[i] for i in range(NUMW)] + [1.0]
         
            ## Input row j of A 
            task.putavec(
                mosek.accmode.con, # Input rows of A. 
                j,             # row index 
                range(NUMVAR), # Column index of non-zeros in column j. 
                aval,          # Non-zero Values of row j. 
                )
        
            ## set bounds on constraints
            task.putbound(
                mosek.accmode.con,
                j,
                mosek.boundkey.lo,
                avg_loss,
                +inf,
                )

       
        # Input the objective sense (minimize/maximize) 
        task.putobjsense(mosek.objsense.minimize)
        
        # import pdb; pdb.set_trace()
        
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
        
        # if solsta == mosek.solsta.optimal or \
            # solsta == mosek.solsta.near_optimal: 
            # print("Optimal solution: %s" % xx) 
        # elif solsta == mosek.solsta.dual_infeas_cer: 
            # print("Primal or dual infeasibility.\n") 
        # elif solsta == mosek.solsta.prim_infeas_cer: 
            # print("Primal or dual infeasibility.\n") 
        # elif solsta == mosek.solsta.near_dual_infeas_cer: 
            # print("Primal or dual infeasibility.\n") 
        # elif solsta == mosek.solsta.near_prim_infeas_cer: 
            # print("Primal or dual infeasibility.\n") 
        # elif mosek.solsta.unknown: 
            # print("Unknown solution status") 
        # else: 
            # print("Other solution status")
        
        w,xi = xx[:NUMW], xx[-1]
        
        return w,xi
 
    def solve_mosek_scale(self, W, gtpsis, w0, **kwargs):
        def streamprinter(text): 
            sys.stdout.write(text) 
            sys.stdout.flush()
            dir  = logging.LOG_OUTPUT_DIR
            rank = logging.RANK
            if dir is not None:
                f = open('{}/output{}.log'.format(dir, rank),'a') 
                f.write(text)
                f.close()
        
        inf = 0
        
        # Open MOSEK and create an environment and task 
        # Make a MOSEK environment 
        env = mosek.Env () 
        
        # Attach a printer to the environment 
        env.set_Stream (mosek.streamtype.log, streamprinter) 
        
        # Create a task 
        task = env.Task() 
        task.set_Stream (mosek.streamtype.log, streamprinter)
 
        # problem dimensions
        NUMX   = len(gtpsis)
        NUMPSI = len(gtpsis[0])
        NUMW   = 1
        NUMVAR = NUMW + 1 # +1 for xi
        NUMCON = len(W)
        NUMANZ = NUMCON*NUMVAR
          

        # set const
        task.putmaxnumvar(NUMVAR) 
        task.putmaxnumcon(NUMCON) 
        task.putmaxnumanz(NUMANZ)
       
        task.append(mosek.accmode.con,NUMCON)
        task.append(mosek.accmode.var,NUMVAR)       
        bkx = [mosek.boundkey.lo for i in range(NUMW)] + \
              [mosek.boundkey.lo]
        blx = [ 0.0 for i in range(NUMW)]  + [0.0]
        bux = [ +inf for i in range(NUMW)] + [+inf]
        
        # linear term in objective
        c = [0 for i in range(NUMW)] + [self.C]
        
        # quadratic term in objective
        qsubi = [0]
        qsubj = [0]
        qval = [sum([w0[i]**2 for i in range(NUMPSI)])]
        
        ## constant term
        task.putcfix(0.0)
        
        # Set the bounds on variable j 
        # blx[j] <= x_j <= bux[j] 
        task.putboundlist(mosek.accmode.var,range(NUMVAR),bkx,blx,bux)
        
        # Set the linear term c_j in the objective. 
        task.putclist(range(NUMVAR),c)
        
        # Set up and input quadratic objective
        task.putqobj(qsubi,qsubj,qval)

        ## psi(x,z)
        avg_psi_gt = [0 for i in range(NUMPSI)]
        for gtpsi in gtpsis:
            for i, p in enumerate(gtpsi):
                avg_psi_gt[i] += 1.0/ float(NUMX) * p

        ## set the constraints
        for j in range(NUMCON): 
            # average psi
            avg_psi = [0 for i in range(NUMPSI)]
            for psi in W[j]['psis']:
                for i,p in enumerate(psi):
                    avg_psi[i] += 1.0/float(NUMX) * p
                    
            # average loss
            avg_loss = 0
            for loss in W[j]['losses']:
                avg_loss += 1.0/float(NUMX) * loss
            
            ## psi(x,y_) - psi(x,z)
            aval = \
                [sum([w0[i]*(avg_psi[i] - avg_psi_gt[i]) for i in range(NUMPSI)])] + [1.0]
         
            ## Input row j of A 
            task.putavec(
                mosek.accmode.con, # Input rows of A. 
                j,             # row index 
                range(NUMVAR), # Column index of non-zeros in column j. 
                aval,          # Non-zero Values of row j. 
                )
        
            ## set bounds on constraints
            task.putbound(
                mosek.accmode.con,
                j,
                mosek.boundkey.lo,
                avg_loss,
                +inf,
                )

       
        # Input the objective sense (minimize/maximize) 
        task.putobjsense(mosek.objsense.minimize)
        
        # import pdb; pdb.set_trace()
        
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
        
        # if solsta == mosek.solsta.optimal or \
            # solsta == mosek.solsta.near_optimal: 
            # print("Optimal solution: %s" % xx) 
        # elif solsta == mosek.solsta.dual_infeas_cer: 
            # print("Primal or dual infeasibility.\n") 
        # elif solsta == mosek.solsta.prim_infeas_cer: 
            # print("Primal or dual infeasibility.\n") 
        # elif solsta == mosek.solsta.near_dual_infeas_cer: 
            # print("Primal or dual infeasibility.\n") 
        # elif solsta == mosek.solsta.near_prim_infeas_cer: 
            # print("Primal or dual infeasibility.\n") 
        # elif mosek.solsta.unknown: 
            # print("Unknown solution status") 
        # else: 
            # print("Other solution status")
        
        _w,xi = xx[0], xx[-1]
        w = [w0[i]*_w for i in range(NUMPSI)] 
        return w,xi

    def no_mosek_solve(self, W, gtpsis, **kwargs):
        from solver_qp_constrained import ObjectiveAPI, ConstrainedSolver
        
        n = len(gtpsis[0])
        ncons = len(W)
        Ssize = len(gtpsis)

        ## psi(x,z)
        avg_psi_gt = np.zeros(n)
        for psi in gtpsis:
            for i_p,p in enumerate(psi):
                avg_psi_gt[i_p] += 1.0/ float(Ssize) * p
        
        def compute_avg_psi(j):
            avg_psi = np.zeros(n)
            for psi in W[j]['psis']:
                for i_p,p in enumerate(psi):
                    avg_psi[i_p] += 1.0/ float(Ssize) * p
            return avg_psi
            
        def compute_avg_loss(j):
            avg_loss = 0
            for loss in W[j]['losses']:
                avg_loss += \
                    1. / float(Ssize) * loss
            return avg_loss
            
        ## objective
        P = np.mat(np.diag([1 for i in range(n)] + [0]))
        q = np.mat(np.c_[[0 for i in range(n)]   + [self.C]])
        G = np.bmat([
            # constraints
            [[[p - pgt for p,pgt in  zip(compute_avg_psi(j),avg_psi_gt)] + [1.] \
                for j in range(ncons)]],
            # positivity constraints on w and xi
            [np.diag([1 for i in range(n)] + [1])],
            ])
        h = np.mat(np.c_[
            # constraints
            [compute_avg_loss(j) for j in range(ncons)] + \
            # positivity constraints
            [0 for i in range(n+1)]
            ])
            
        ## make solver object
        obj = ObjectiveAPI(P,q,G=G,h=h)
        solver = ConstrainedSolver(obj,epsilon=1e-15,t0=1.0)
        
        ## make initial guess for w,xi
        if w is None:
            w = np.zeros(n)

        w_ = np.asarray(w) + 1e-5
        xi0 = np.maximum(np.max(h[:ncons] - G[:ncons,:-1]*np.mat(w_).T),0) + 1e-5
        sol0 = np.mat(np.r_[w_, xi0]).T
       
        ## solve w,xi
        sol = solver.solve(sol0,epsilon=1e-8) 
        
        ## return solution
        w,xi = sol[:n].A.ravel().tolist(), float(sol[n].A)
        return w, xi
            
 
