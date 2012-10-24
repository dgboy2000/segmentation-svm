import numpy as np
from rmn.fast_marching import fastm

if __name__=='__main__':    
    
    shape = (100,100)
    speed = np.random.random(shape)
    
    npoints = 10
    points = np.c_[
        np.random.randint(0, shape[0], npoints),
        np.random.randint(0, shape[1], npoints),
        ]
    
    dist, labels, DM, inter_ = fastm.fast_marching_3d(
        speed, 
        points,
        offset=1,
        output_arguments=(
            'distances', 
            'labels', 
            'DM',
            'intersections',
            )
        )
    inter = np.argwhere(inter_!=-1)
    
    path = fastm.shortestpath(points, inter_, dist)
    path_ = np.argwhere(path!=-1)
    col = path[tuple(path_.T)]
    
    if 1:
        from matplotlib import pyplot
        pyplot.figure()
        pyplot.imshow(speed.T, cmap='gray', aspect='equal', interpolation='nearest')
        pyplot.imshow(labels.T,  alpha=0.8, interpolation='nearest')
        pyplot.scatter(points[:,0], points[:,1])
        pyplot.scatter(inter[:,0], inter[:,1], c='r')
        pyplot.scatter(path_[:,0], path_[:,1], c=col)
        pyplot.show()