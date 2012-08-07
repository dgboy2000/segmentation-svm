import logging

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
        self.epsilon = kwargs.pop('epsilon',1e-5)
        
        self.loss = loss_function
        self.psi  = psi
        self.mvc  = most_violated_constraint
        
        self.classifier = None
        
        # create logger with 'spam_application'
        logger = logging.getLogger('svm logger')
        loglevel = kwargs.pop('loglevel',logging.WARNING)
        logger.setLevel(loglevel)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(ch)
    
        self.logger = logger
    
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
                avg_phi_gt[i_p] += 1.0/ float(Ssize) * p
        
        ## set the constraints
        for j in range(NUMCON): 
         
            ## psi(x,y_)
            avg_phi = [0 for i in range(self.wsize)]

            avg_loss = 0
            for i_y,y_ in enumerate(W[j]):
                # average loss
                avg_loss += \
                    1. / float(Ssize) *\
                    self.loss(self.S[i_y][1], y_) 
                
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
        
        w,xi = xx[:self.wsize], xx[-1]
        
        return w,xi
        
    def _stop_condition(self,w,xi,ys):
        cond = 0
        for s,y_ in zip(self.S,ys):
            x,z = s
            cond += self.loss(z,y_)
            psi_xy_ = self.psi(x,y_)
            psi_xz  = self.psi(x,z)
            for i in range(len(w)):
                cond -= w[i]*(psi_xy_[i] - psi_xz[i])
        
        cond /= float(len(self.S))
        
        if cond <= xi + self.epsilon:
            return True
        else:
            return False
            
            
    def train(self, verb=0):
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
        
        niter = 1
        while 1:
            ## compute current solution (qp + constraints)
            self.logger.info("compute current solution")
            w,xi = self._current_solution(W)
            self.logger.debug("w={}, xi={:.2}".format(w,xi))
        
            ## find most violated constraint
            self.logger.info("find most violated constraint")
            ys = []
            for s in self.S:
                y_ = self.mvc(w, *s)
                ys.append(y_)
            self.logger.debug("ys={}".format(ys))
            
            ## add to test set
            W.append(ys)
            
            ## stop condition
            if self._stop_condition(w,xi,ys): 
                self.logger.info("stop condition reached")
                break
            else: niter+= 1
            
            self.logger.debug("iteration #{}".format(niter))
        
        ## return values
        info = {
            'number of iterations': niter, 
            'number of contraints': len(W),
            }
        
        for msg in info: self.logger.info("{}={}".format(msg,info[msg]))
            
        return w, xi, info

        
if __name__=='__main__':

    import random
    
    ## test svm struct
    
    ## training set: N dimensionnal Gaussians
    K = 1000 # number of training examples
    L = 5    # number of classes
    N = 10   # feature size
    
    sigma = 1e-2 # sigma for the gaussians
    
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
            return min(scores)[1]
            
    my_mvc = My_mvc(L)
            
    ## run svm struct
    svm = StructSVM(
        S, 
        my_loss, 
        my_psi,
        my_mvc, 
        C=100, 
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
    