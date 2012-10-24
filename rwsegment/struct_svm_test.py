import numpy as np
from struct_svm import *

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
        
    
