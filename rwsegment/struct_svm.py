import logging
import numpy as np

import mpi

import utils_logging
logger = utils_logging.get_logger('struct_svm',utils_logging.DEBUG)

class DataContainer(object):
    def __init__(self, data):
        self.data = data
    def __hash__(self):
        return id(self.data)


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
        
        S = []
        for x,z in training_set:
            S.append((DataContainer(x), DataContainer(z)))
        self.S = S
        
        self.C = kwargs.pop('C',1.)
        self.epsilon = kwargs.pop('epsilon',1e-5)
        self.nitermax = kwargs.pop('nitermax',100)
        
        self.wsize = kwargs.pop('wsize',None)
        
        self.user_loss = loss_function
        self.user_psi  = psi
        self.user_mvc  = most_violated_constraint 
       
        self.psi_cache  = {}
        self.psis_cache = {}
        self.loss_cache = {}
        self.mvc_cache  = {}      
  
        self.psi_scale = kwargs.pop('psi_scale', 1.0)
        self.use_parallel = kwargs.pop('use_parallel', False)
        nomosek = kwargs.pop('nomosek',False)

        self.do_switch_loss = kwargs.pop('do_switch_loss', False)

        try:
            if nomosek:
                 self._current_solution = self.no_mosek_current_solution
            else:
                import mosek
                self._current_solution = self.mosek_current_solution
        except ImportError:
            logger.warning('Mosek not found, using solver_qp_constrained')
            self._current_solution = self.no_mosek_current_solution
        
        
    def parallel_mvc(self,w, **kwargs):
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        comm = mpi.COMM
        size = mpi.SIZE
        
        opts = kwargs         

        ntrain = len(self.S)
        indices = np.arange(ntrain)
        for n in range(1,size):       
            inds = indices[np.mod(indices,size-1) == (n-1)]
            comm.send(('mvc',len(inds),opts), dest=n)
            for i in inds:
                x,z = self.S[i]
                comm.send((i,w,x,z), dest=n)

        ys = []
        ntrain = len(self.S)
        for i in range(ntrain):
            source_id = np.mod(i,comm.Get_size()-1) + 1
            ys.append( 
                comm.recv(source=source_id,tag=i),
                )
        return ys
    
    
    def parallel_all_psi(self,ys=None):
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        comm = mpi.COMM
        size = mpi.SIZE
       
        opts = {}
  
        ## send training data and cutting plane
        ntrain = len(self.S)
        indices = np.arange(ntrain)
        for n in range(1,size):       
            inds = indices[np.mod(indices,size-1) == (n-1)]
            comm.send(('psi',len(inds), opts), dest=n)
            for ind in inds:
                x,z = self.S[ind]
                if ys is None:                
                    comm.send((ind,x,z), dest=n)
                else:
                    comm.send((ind,x,ys[ind]), dest=n)
    
    
        ## get the psis back
        cond = 0
        ntrain = len(self.S)
        psis = []
        for i in range(ntrain):          
            ## recieve from workers
            source_id = np.mod(i,comm.Get_size()-1) + 1
            psi = comm.recv(source=source_id,tag=i)
            psis.append(psi)
            
        # return psis
        for i in range(ntrain):
            yield psis[i]
    

    
    def no_mosek_current_solution(self,W, w=None, xi=None):
        from solver_qp_constrained import ObjectiveAPI, ConstrainedSolver
        
        n = self.wsize
        ncons = len(W)
        
        if ncons == 0:
            w = [1.0 for i in range(n)]
            xi = 0.0
            return w,xi
        
        ## psi(x,z)
        Ssize = len(self.S)
        avg_psi_gt = np.zeros(n)
        for psi in self.compute_all_psi():
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
        P = np.mat(np.diag([s for s in self.psi_scale] + [0]))
        q = np.mat(np.c_[[0 for i in range(n)] + [self.C]])
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
        return w,xi
    
    def mosek_current_solution(self, W, **kwargs):
        ''' quadratic programming 
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
        
        if len(W)==0:
            w = [1.0 for i in range(self.wsize)]
            xi = 0.0
            return w,xi
        
        import mosek
        import sys
        
        mosek.iparam.log = 0
        
        def streamprinter(text): 
            sys.stdout.write(text) 
            sys.stdout.flush()
        
        inf = 0
        
        # Open MOSEK and create an environment and task 
        # Make a MOSEK environment 
        env = mosek.Env () 
        
        # Attach a printer to the environment 
        # env.set_Stream (mosek.streamtype.log, streamprinter) 
        
        # Create a task 
        task = env.Task() 
        # task.set_Stream (mosek.streamtype.log, streamprinter)
        
        # Set up and input bounds and linear coefficients
        # bkx = [mosek.boundkey.fr for i in range(self.wsize)] + \
              # [mosek.boundkey.lo]
        # blx = [ -inf for i in range(self.wsize)] + [0.0]
        bkx = [mosek.boundkey.lo for i in range(self.wsize)] + \
              [mosek.boundkey.lo]
        blx = [ 0.0 for i in range(self.wsize)] + [0.0]
        
        bux = [ +inf for i in range(self.wsize)] + [+inf]
        
        c = [0 for i in range(self.wsize)] + [self.C]
        
        qsubi = range(self.wsize)
        qsubj = qsubi
        # qval = [1.0]*self.wsize
        qval = (self.psi_scale * np.ones(self.wsize)).tolist()
        
        NUMVAR = len(bkx) 
        NUMCON = len(W)
        NUMANZ = NUMCON*NUMVAR
        
        task.putmaxnumvar(NUMVAR) 
        task.putmaxnumcon(NUMCON) 
        task.putmaxnumanz(NUMANZ)
        
        task.append(mosek.accmode.con,NUMCON)
        task.append(mosek.accmode.var,NUMVAR)
        task.putcfix(0.0)
        
        ## psi(x,z)
        Ssize = len(self.S)
        avg_psi_gt = [0 for i in range(self.wsize)]
        for psi in self.compute_all_psi():
            for i_p,p in enumerate(psi):
                avg_psi_gt[i_p] += 1.0/ float(Ssize) * p
             
        
        ## set the constraints
        for j in range(NUMCON): 
            
            ## psi(x,y_)
            # average psi
            avg_psi = [0 for i in range(self.wsize)]
            for psi in W[j]['psis']:
                for i_p,p in enumerate(psi):
                    avg_psi[i_p] += 1.0/ float(Ssize) * p
                    
            # average loss
            avg_loss = 0
            for loss in W[j]['losses']:
                avg_loss += \
                    1. / float(Ssize) * loss
            
            ## psi(x,y_) - psi(x,z)
            aval = \
                [avg_psi[i] - avg_psi_gt[i] for i in range(self.wsize)] + \
                [1.0]
         
            ## Input row j of A 
            task.putavec(
                mosek.accmode.con, # Input rows of A. 
                j,             # Variable (column) index. 
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
        
        # Set the bounds on variable j 
        # blx[j] <= x_j <= bux[j] 
        task.putboundlist(mosek.accmode.var,range(NUMVAR),bkx,blx,bux)
        
        # Set the linear term c_j in the objective. 
        task.putclist(range(NUMVAR),c)
        
        # Set up and input quadratic objective
        task.putqobj(qsubi,qsubj,qval)
        
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
        
        w,xi = xx[:self.wsize], xx[-1]
        
        return w,xi
            
            
    def stop_condition(self,w,xi,ys):
        cond = 0
        ntrain = len(self.S)
        for i,psi_xz,psi_xy_ in zip(
                range(ntrain),
                self.compute_all_psi(),
                self.compute_all_psi(ys)):
            x,z = self.S[i]
            y_ = ys[i]
            cond += self.loss(z,y_)
            for iw in range(len(w)):
                cond -= w[iw]*(psi_xy_[iw] - psi_xz[iw])
        
        cond /= float(len(self.S))
        
        if cond <= xi + self.epsilon:
            return True
        else:
            return False
            
    def psi(self, x,y):
        # return self.user_psi(x.data,y.data)
        if (x,y) in self.psi_cache:
            return self.psi_cache[(x,y)]
        else:
            v = self.user_psi(x.data,y.data)
            self.psi_cache[(x,y)] = v
            return v
            
    def compute_all_psi(self, ys=None):
        if ys is None:
            obj = None
        else:
            obj = ys[0]
        if hash(obj) in self.psis_cache:
            for psi in self.psis_cache[hash(obj)]:
                yield psi
            
        elif self.use_parallel:
            self.psis_cache[hash(obj)] = []
            for psi in self.parallel_all_psi(ys):
                self.psis_cache[hash(obj)].append(psi)
                yield psi
        else:
            self.psis_cache[hash(obj)] = []
            ntrain = len(self.S)
            for i in range(ntrain):
                x,z = self.S[i]
                if ys is None:
                    psi = self.psi(x,z)
                else:
                    psi = self.psi(x,ys[i])
                    
                self.psis_cache[hash(obj)].append(psi)
                yield psi
            
    def loss(self,z,y):
        return self.user_loss(z.data,y.data)
        
    def mvc(self,w,x,z, **kwargs):
        return DataContainer(self.user_mvc(w,x.data,z.data,**kwargs))
            
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
        w,xi = None,None
       
        ## compute psis of ground truth
        logger.debug('compute psis of ground truth')
        gtpsis = list(self.compute_all_psi())
        
        ## log psis
        strpsi = [' '.join('{:.3}'.format(val) for val in psi) for psi in gtpsis]
        logger.debug('ground truth psis: {}'.format(strpsi))
 
        ## initialize w
        if self.wsize is None:
            logger.info("compute length of psi")
            self.wsize = len(self.psi(*self.S[0]))
       
        switch_loss = False 
        niter = 1
        while 1:
            logger.info("iteration (struct) #{}".format(niter))
            
            # garbage collect to save memory on each iteration
            import gc
            gc.collect()
            
            ## compute current solution (qp + constraints)
            logger.info("compute current solution")
            w,xi = self._current_solution(W,w=w,xi=xi)
            
            ## logging
            wstr = ' '.join('{:.2}'.format(wval) for wval in w)
            logger.debug("w=[{}], xi={:.2}".format(wstr,xi))
            
            wscaled = np.asarray(w)*self.psi_scale
            wsstr = ' '.join('{:.2}'.format(wval) for wval in wscaled)
            logger.debug("scaled w=[{}]".format(wsstr))
            
            objective = 0.5*np.dot(w,w) + self.C*xi
            logger.debug("objective={}".format(objective))
        
        
            ## find most violated constraint
            logger.info("find most violated constraint")
            ys = []
            
            if self.use_parallel:
                ys = self.parallel_mvc(w, switch_loss=switch_loss)
            else:
                for s in self.S:
                    import ipdb; ipdb.set_trace()
                    x,z = s
                    y_ = self.mvc(w, x, z, exact=True, switch_loss=switch_loss)
                    ys.append(y_)
                    if np.std(np.sum(y_.data,axis=0)) > 1e-5:
                        import ipdb; ipdb.set_trace()
            
            ## compute psis and losses:
            logger.debug('compute psis and losses for added constraints')
            psis = list(self.compute_all_psi(ys))
            losses = [self.loss(self.S[i][1], y_) for i,y_ in enumerate(ys)]
            
            ## log psis and losses
            strpsi = [' '.join('{:.3}'.format(val) for val in psi) for psi in psis]
            logger.debug('new psis: {}'.format(strpsi))
            strloss = ' '.join('{:.3}'.format(val) for val in losses)
            logger.debug('new losses: {}'.format(strloss))
            
            ## add to test set
            W.append({'psis': psis, 'losses': losses})
            
            ## stop condition
            logger.debug('compute stop condition')
            if self.stop_condition(w,xi,ys):
                if self.do_switch_loss:
                    logger.info("stop condition reached, switching loss")
                    self.do_switch_loss = False
                    switch_loss = True
                else:     
                    logger.info("stop condition reached, stopping")
                    break
            elif niter >= self.nitermax:
                logger.info("max number of iterations reached")
                break

            ## continuing
            niter+= 1

        
        ## return values
        info = {
            'number of iterations': niter, 
            'number of contraints': len(W),
            }
        
        for msg in info: logger.info("{}={}".format(msg,info[msg]))
            
        return w, xi, info

        
if __name__=='__main__':

    import random
    
    ## test svm struct
    
    ## training set: N dimensionnal Gaussians
    K = 500  # number of training examples
    L = 5    # number of classes
    N = 10   # feature size
    
    sigma = 1e0 # sigma for the gaussians
    
    # 1 - generate centers
    G = []
    for l in range(L):
        G.append([random.random()*10 for n in range(N)])
        
    # 2 - generate training set
    S = []
    for k in range(K):
        z = random.randint(0,L-1)
        x = [random.gauss(g,sigma) for g in G[z]]
        S.append((x,z))
    
    ## specific functions for struct svm learning
    def my_loss(z,y_):
        if y_!= z: return 1
        else: return 0
    
    class My_psi(object):
        def __init__(self,nclass):
            self.nclass = nclass
        def __call__(self,x,y):
            v = []
            for n in range(self.nclass):
                if y==n: v += [val for val in x]
                else:    v += [0 for v in x]
            return v
                
    my_psi = My_psi(L)
    
    
    class My_mvc(object):
        ## most violated constraint
        def __init__(self,nclass):
            self.nclass = nclass
           
        def _score(self,w,x,y):
            n = len(w)
            return sum([w[i]*my_psi(x,y)[i] for i in range(n)])
            
        def __call__(self,w,x,z):
            scores = [
                (self._score(w,x,y_) - my_loss(z,y_), y_)\
                for y_ in range(self.nclass)
                ]
            #import pdb; pdb.set_trace()
            return min(scores)[1]
            
    my_mvc = My_mvc(L)
            
    ## run svm struct
    svm = StructSVM(
        S, 
        my_loss, 
        my_psi,
        my_mvc, 
        C=100, 
        epsilon=1e-3,
        nitermax=100,
        loglevel=logging.INFO,
        )
    w,xi,info = svm.train()
    
    class My_classifier(object):
        def __init__(self,w,classes):
            self.w = w
            self.classes = classes
        def __call__(self,x):
            scores = [
                (sum([w[i]*my_psi(x,s)[i] for i in range(len(w))]),s)\
                for s in self.classes
                ]
            y = min(scores)[1]
            return y
    my_classifier = My_classifier(w,range(L))
    
    class_train = [(s[1],my_classifier(s[0])) for s in S]
    print 'mis-classified training samples:'
    for e in class_train:
        if e[0]!=e[1]: print e
        
    
    
