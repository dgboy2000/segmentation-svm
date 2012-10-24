import numpy as np

from rmn.algebra import spectralg, distance_matrix as distm

pt1 = [
    [0,0],
    [0,1],
    [1,0],
    ]
pt2 = [
    [0.1,0.2],
    [-0.2,0.05],
    [0.0,0.1],
    [-0.1,0.97],
    [0.2,1.1],
    [1.2,0.0],
    ]

DM1 = distm.euclidian(pt1)
DM2 = distm.euclidian(pt2)    
    

A1 = 1./(1e-10 + DM1)
A2 = 1./(1e-10 + DM2)

w1,V1 = spectralg.laplacian_eigen_maps(A1,symmetric=True)
w2,V2 = spectralg.laplacian_eigen_maps(A2,symmetric=True)