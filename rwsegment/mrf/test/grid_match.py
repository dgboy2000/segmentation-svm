import numpy as np
from scikits import delaunay
from scipy import comb as combination
from matplotlib import pyplot

from rmn.algebra import distance_matrix as distm
from rmn.graph import fastPD, trw



def plot(pt1, pt2, matches, labels=None, pairs=None, figure=None):

    npt1 = len(pt1)
    npt2 = len(pt2)
    
    if figure is None: figure = pyplot.figure()
    
    if labels is None:
        colors = np.arange(npt2)
    else:
        colors = labels
    
    pyplot.scatter(
        pt2[:,0],pt2[:,1],
        c=colors,
        s=100,
        marker='s',
        figure=figure,
        )
    
    pyplot.scatter(
        pt1[:,0],pt1[:,1],
        c=colors[matches],
        s=50,
        marker='+',
        figure=figure,
        )
            
    pyplot.quiver(
        pt1[:,0],pt1[:,1],
        pt2[matches,0]-pt1[:,0],
        pt2[matches,1]-pt1[:,1],
        angles='xy', scale_units='xy', scale=1,
        color='r',
        figure=figure,
        width=0.002,
        )
    
    
    if pairs is not None:
        pyplot.quiver(
            pt1[pairs[:,0],0],pt1[pairs[:,0],1],
            pt1[pairs[:,1],0]-pt1[pairs[:,0],0],
            pt1[pairs[:,1],1]-pt1[pairs[:,0],1],
            angles='xy', scale_units='xy', scale=1,
            figure=figure,
            width=0.001, 
            alpha='0.1',
            )
            
    pyplot.grid(b=True)
    pyplot.show()

    
    
if __name__=='__main__':
    
    grid_shape = (4,5)
    step_size = 3
    nnodes1 = 60
    nnodes2 = np.prod(grid_shape)
    
    print '---------------------------------------------------------'
    print '\ntest graph matching on a %dx%d grid' %(nnodes1,nnodes2)
    
    if 'pt2' not in dir() or 'pt1' not in dir() or \
            len(pt2)!=nnodes2 or len(pt1)!=nnodes1:
        print 'compute points'
        pt2 = np.argwhere(np.ones(grid_shape))*step_size
        gt = np.random.randint(nnodes2, size=nnodes1)
        pt1 = pt2[gt,:] + np.random.random((nnodes1,2))*2 - 1
    
    use_permuted_ordering = False
    if use_permuted_ordering:
        print 'use permuted ordering'
        ordering = (np.tile(range(nnodes2),(nnodes1,1)) + \
                np.c_[np.arange(nnodes1)])%nnodes2
    else:
        ordering = np.tile(range(nnodes2),(nnodes1,1))
    gt_ = np.where((ordering-np.c_[gt])==0)[1]
    
    print '\ncompute unary cost'
    nlabels = 1 #nnodes2
    ucost = 100
    
    print 'unary cost = %d for %d distinct labels' %(ucost,nlabels)
    labels = np.arange(nnodes2)%nlabels
    unary = (ordering%nlabels != np.tile(np.c_[gt%nlabels],(1,nnodes2)))*ucost
    
    use_proximity = True
    if use_proximity:
        thresh = step_size
        print 'use proximity with thresh = %f' %thresh
        DM_ = distm.euclidian(pt1,pt2)# - thresh
        # DM_[DM_<0] = 0
        unary += DM_/thresh + np.power(DM_/thresh,3.)
    
    print '\ncompute binary costs'
    if 'connectivity' not in dir():
        connectivity = 'full'  #'delaunay', 'full'
    
    if connectivity=='delaunay':
        print '(delaunay connected graph)'
        pairs = delaunay.delaunay(pt1[:,0],pt1[:,1])[1]
        
    elif connectivity=='smallworld':
        prop = 1
        print '(small world connected graph with %d pc random edges added)' %prop
        pairs = delaunay.delaunay(pt1[:,0],pt1[:,1])[1]
        nadded = int(combination(nnodes1,2,exact=1)*prop/100. + 0.5)
        newpairs = np.random.randint(0,nnodes1,size=(nadded,2))
        newpairs = newpairs[newpairs[:,0]!=newpairs[:,1]]
        pairs = np.r_[pairs,newpairs]
        
    else:
        print '(fully connected graph)'
        pairs = np.argwhere(np.triu(np.ones((nnodes1,nnodes1)),k=1))
   
    DM1 = distm.euclidian(pt1)
    DM2 = distm.euclidian(pt2)
    
    def cost_function(ie,l1,l2):
            n1,n2 = pairs[ie]
            
            cost = np.abs(DM1[n1,n2] - DM2[ordering[n1,l1],ordering[n2,l2]])
            return cost + cost**4
    
    if 'solver' not in dir(): solver = 'trw' # 'fastPD'
    
    if solver == 'fastPD':
        
        print '\ncompute solution with fastPD'
        solution0, energy = fastPD.fastPD_callback(
            unary,
            pairs,
            cost_function,
            niters=10000,
            )
        solution = ordering[range(nnodes1),solution0]
            
        print '\nsolution energy'
        fastPD.compute_energy_callback(
            solution0, unary, pairs, cost_function, display=True)
            
        print '\nground truth energy'
        fastPD.compute_energy_callback(
            gt_, unary, pairs, cost_function, display=True)
    else:
        wpairs = np.zeros((len(pairs),nnodes2*nnodes2))
        pairs2 = np.where(np.ones((nnodes2,nnodes2)))
        for ip in range(len(pairs)):
            wpairs[ip,:] = cost_function(
                ip,
                *pairs2
                )
                # np.repeat(ordering[pairs[ip,0],:],nnodes2),
                # np.tile(ordering[pairs[ip,1],:],nnodes2),
                # )
            
        print '\ncompute solution with TRW'
        solution0, energy = trw.TRW_general(
            unary,
            pairs,
            wpairs,
            niters=3000,
            use_bp=False,
            randomize=True,
            )
        solution = ordering[range(nnodes1),solution0]
        
        print '\nsolution energy'
        trw.compute_energy(
            solution0, unary, pairs, wpairs, display=True)
            
        print '\nground truth energy'
        trw.compute_energy(
            gt_, unary, pairs, wpairs, display=True)
    
    plot(pt1, pt2, solution, pairs=pairs,labels=labels)
    
    
    