import numpy as np
from matplotlib import pyplot


from rmn.fast_marching import fastm
from rmn.algebra import distance_matrix
from rmn.filter import fft
from rmn.geometry import delaunay_2d

from rmn.algebra import random_walker
from rmn.algebra.commute_time import commute_time

from geodesic_matching import plot_graph

if __name__=='__main__':
    
    # make image
    im = np.zeros((102, 155))
    im[:,50] = 1
    im[:51,-50] = 1
    
    im += np.random.randn(*im.shape)/10
    
    K = fft.gaussian_kernel(2, ndim=2)
    localstd = np.sqrt(fft.filter(im**2,K) - fft.filter(im,K)**2)
    imw = localstd/(localstd+np.std(im))


    #points
    points = np.asarray([
        [25, 25],
        [25, 77],
        [25, 129],
        [76, 25],
        [76, 77],
        [76, 129],
        ])
    
    _ipoints = np.unique(np.random.randint(0, np.prod(im.shape), 200))
    rpoints = np.c_[
        np.asarray(np.indices(im.shape))[0,:,:].flat[_ipoints],
        np.asarray(np.indices(im.shape))[1,:,:].flat[_ipoints],
        ]
    
    # pyplot.figure()
    # pyplot.imshow(imw.T, cmap='gray')
    # pyplot.scatter(rpoints[:,0], rpoints[:,1])
    # pyplot.show()
    
        
    ipoints = np.argmin(distance_matrix.euclidian(points, rpoints), axis=1)
    npoints= len(ipoints)
    


    # geodesic distances
    rE = distance_matrix.euclidian(rpoints)
    rC = delaunay_2d.connectivity(rpoints)
    rG,rparents = fastm.fast_marching_3d_aniso(
        # im*np.mean(rE[rC==1]),
        imw,
        rpoints,
        heap_size=1e6,
        offset=1e-5,
        connectivity26=True,
        output_arguments=('DM','parents'),
        )
    pathes,connections = fastm.shortestpath(rparents)
    impath = np.zeros(imw.shape)
    for path in pathes:
        impath[tuple(pathes[path].T)] = path
        
if 0:  
    # rM = rE*(rG<1e9)/(rG*(rG<1e9) + (rG>1e9))
    # rM = (rG >= 3*rE)
    rM = np.minimum(np.exp((rG - 3*rE)/10),1)
    
    # Mcom = commute_time(1-rM)[ipoints,:][:,ipoints]
    # Mcom = Mcom - np.min(Mcom[np.where(np.triu(np.ones(Mcom.shape),k=1))])
    # Mcom.flat[::Mcom.shape[0]+1] = 0
    # Mcom /= np.c_[np.sum(Mcom, axis=1)]
    
    Aff = np.exp(-rG**2/np.mean(rG[rG<1e9]*3)**2)
    Mrw = random_walker.rw(1-rM, ipoints) - np.eye(npoints)
    # Mrw = random_walker.rw(Aff, ipoints) - np.eye(npoints)

    # euclidian distances
    E = distance_matrix.euclidian(points)
    
    # geodesic distances
    G, dist = fastm.fast_marching_3d_aniso(
        # (im-np.mean(im))/np.std(im),
        # im,
        # im*np.mean(E)/2,
        im*np.mean(E)/2,
        points,
        heap_size=1e6,
        offset=1,
        connectivity26=True,
        output_arguments=('DM','distances'),
        )
    M = np.minimum(np.exp((G - 3*E)/10),1)
    
    # Me = (Mrw-np.eye(6))/(E + np.eye(6))
    # Me = Me/np.sum(Me, axis=1)
    Me = Mrw*E
    Me = 1 - Me/np.c_[np.sum(Me, axis=1)]
    
    # plot
    fig = pyplot.figure()
    pyplot.imshow(im.T, cmap='gray', interpolation='nearest', figure=fig)
    # pyplot.imshow(dist.T, interpolation='nearest', alpha=0.5, figure=fig)
    # pyplot.scatter(points[:,0], points[:,1], figure=fig)
    # pyplot.show()
    
    # plot_graph.plot(rpoints, E=rM, connectivity=(rG<1e9), img=im, text_edges=True)
    # plot_graph.plot(points, E=M, connectivity=(G<1e9), img=im, text_edges=True)
    # plot_graph.plot(points, E=Mcom, connectivity=(G<1e9), img=im, text_edges=True)
    # plot_graph.plot(
        # points, E=Mrw, connectivity=(G<1e9), img=im, text_edges=True, text_nodes=True)
    
    plot_graph.plot(
        points, E=(Mrw+Mrw.T)/2, connectivity=(G<1e9), img=im, text_edges=True, text_nodes=True)
    
        
    