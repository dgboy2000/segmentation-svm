import numpy as np
from scipy.misc import comb


class FFD(object):
    ''' n-dimensional FFD
    
        usage: 
        
            # initialize ffd to a certain size and shape
            ffd = FFD(extent, shape=grid_shape, degree=3)
            
            # set initial point cloud
            ffd.set_points(pts)
            
            # move control points
            control_pts = np.copy(ffd.controls)
            ... (move control_pts)
            
            # get deformed point cloud
            deformed_pts = ffd.deform(control_pts)
            
        note:
        
            - extent = (x_min, y_min, ..., x_max, y_max, ...)
            - shape = n or (nx, ny, ...)  # number of controls per axis
            - degree = d or (dx, dy, ...) # degree of the FFD per axis
            
        
    
        A FFD cell in 3d:
         
            1          3
        (x) +----------+
           /:         /|
         0/ : (y)   2/ |
         +----------+  |
         |  +.......|..+
         | ,5       | /7
     (z) |,         |/
         +----------+
         4          6
    
        order of control points in a cell: 
            [( i%2, (i/2)%2, ..., (i/2^(ndim-1))%2 for i in range(2^ndim)]
    
    '''
    
    def __init__(self, extent, shape=2, degree=3):
        
        ndim = len(extent)/2
        bbox_min = np.asarray(extent)[:ndim]
        bbox_max = np.asarray(extent)[ndim:]
        
        self.degree = np.ones(ndim, dtype=int)*int(degree)
        self.ndim = ndim
        self.extent = extent
        self.shape = np.ones(ndim)*shape
        self.spacing = (bbox_max-bbox_min)/(self.shape - 1)
        
        # generate control points
        controls, edges, cells = self._regular_lattice(self.shape)
        controls = controls/(self.shape - (shape>1)*1).astype(float)
        controls = controls*(bbox_max-bbox_min) + bbox_min

        self.controls = controls
        self.edges = edges
        self.cells = cells
        
        # generate FFD lattice
        self.make_lattice()
        
        # if not points is None: 
            # self.set_points(points)
        
        
        
    def make_lattice(self):
        # generate inner grid (according to the degree)
        gshape = self.degree + 1
        ndim = self.ndim
        
        icontrols = np.r_[
            [np.mod(i/2**np.arange(ndim),2) for i in range(2**ndim)]
            ]
        
        pows = np.indices(gshape).ravel().reshape((ndim,-1)).T
        grid = pows/(gshape-1).astype(float)

        self.icontrols = icontrols
        self.pows = pows
        self.grid = grid

    def setpoints(self, points, use_clib=False):
        
        self.points0 = points
    
        if use_clib:
            ## use ctypes library   
            
            if not self.ndim==3:
                print 'only implemted for 3d FFD'
                1/0
                
            ## compute point coordinates
            icells, coords = self.get_coords(self.points0, use_clib=True)

        else:
             icells, coords = self.get_coords(self.points0, use_clib=False)
        
        
        self.coords = coords
        self.icells = icells
            
        
    def deform(self, controls, use_clib=False):
        ''' compute the new position of 'points' given 
        the new position of the control points 
        
        '''

        if use_clib:
            ## use ctypes library   
            
            if not self.ndim==3:
                print 'only implemted for 3d FFD'
                1/0
                
            results = np.zeros(self.points0.shape, dtype=np.float64, order='C')
            libffd.deform_3d(
                len(self.points0),
                np.asarray(self.coords, dtype=np.float64, order='C'),
                np.asarray(self.icells, dtype=np.int32, order='C'),
                len(controls),
                np.asarray(controls, dtype=np.float64, order='C'),
                len(self.cells),
                np.asarray(self.cells, dtype=np.uint32, order='C'),
                np.asarray(self.degree, dtype=np.uint32, order='C'),
                results,
                )
            results[self.icells<0] = self.points0[self.icells<0]
                
        else:
            ## pure python

            ## compute intermediary matrices
            coords = self.coords[:, np.newaxis]
            pows = self.pows[np.newaxis]
            degree = self.degree[np.newaxis, np.newaxis]
            
            ## combinations 
            combs = np.product(comb(self.degree, self.pows), axis=1)
            
            ## ffd coefficients
            W = np.matrix(
                combs* \
                np.product( 
                    np.power(1-coords, degree - pows)*\
                    np.power(coords,   pows),
                    axis=-1,
                    )
                )
            self.W = W
        
            ## compute new point positions
            results = np.copy(self.points0)
            
            for ic,c in enumerate(self.cells):
                i_pts = self.icells==ic
                P = self.get_grid_points(controls[c])
                results[i_pts] = 0
                results[i_pts] = self.W[i_pts]*P
                
        return results
        
        
    def _regular_lattice(self, shape):

        ndim = len(shape)
        nodes = np.indices(shape).ravel().reshape((ndim,-1)).T

        icontrols = np.r_[
            [np.mod(i/2**np.arange(ndim),2) for i in range(2**ndim)]
            ]
            
        inds = np.arange(np.product(shape)).reshape(shape)
        s = np.array([slice(0,-1), slice(1,None)])
        
        cells = None
        for cn in icontrols:
            if cells is None:
                cells = inds[tuple([s[i] for i in cn])].ravel()
            else:
                cells = np.c_[cells, inds[tuple([s[i] for i in cn])].ravel()]
        cells = cells.astype(int)

        s1 = np.array([slice(None), slice(0,-1)])
        s2 = np.array([slice(None), slice(1,None)])
        edges = np.r_[tuple([
            np.c_[
                inds[tuple([s1[1*(d==dd)] for dd in range(ndim)])].ravel(), 
                inds[tuple([s2[1*(d==dd)] for dd in range(ndim)])].ravel(),
                ]
            for d in range(ndim)
            ])]
        edges = edges.astype(int)
        return nodes, edges, cells
        
        
        
    def get_grid_points(self, controls):
        x = self.grid
        if self.ndim==2:
            P = np.dot(np.c_[tuple([v1*v2 \
                for v1 in [1-x[:,[0]], x[:,[0]]]\
                for v2 in [1-x[:,[1]], x[:,[1]]]\
                ])], controls)
        else:
            P = np.dot(np.c_[tuple([v1*v2*v3 \
                    for v1 in [1-x[:,[0]], x[:,[0]]]\
                    for v2 in [1-x[:,[1]], x[:,[1]]]\
                    for v3 in [1-x[:,[2]], x[:,[2]]]\
                    ])], controls)
        return P
        
        
    def get_coords(self, points, **kwargs):
        coords = np.zeros((len(points),self.ndim))
        icells = -np.ones((len(points)), dtype=int)
        
        for icell, cell in enumerate(self.cells):
            _control = self.controls[cell]
            bbox_min = np.min(_control, axis=0)
            bbox_max = np.max(_control, axis=0)
            
            inside = np.all(
                (points >= bbox_min)&\
                (points <  bbox_max), 
                axis=1,
                )
                
            if inside.any():
                icells[inside] = icell
                coords[inside,:] = self.get_coords_cell(points[inside], cell)
                
        return icells, coords
        
        
    def get_coords_cell(self, points, cell):
    
        controls = self.controls[cell][np.sum(self.icontrols, axis=1)<=1]
        
        p = points - controls[0]
        V = controls[1:] - controls[0]
        lV = np.sum(V**2, axis=1)
        
        coords = np.c_[tuple(
            [np.dot(p,V[c]) / lV[c] for c in range(self.ndim)]
            )]
            
        return coords
        
#-------------------------------------------------------------------------------
class EFFD(FFD):
    ''' cocquillard '90 
    
        A FFD cell in 3d:
        
            1          3
        (x) +----------+
           /:         /|
         0/ : (y)   2/ |
         +----------+  |
         |  +.......|..+
         | ,5       | /7
     (z) |,         |/
         +----------+
         4          6
         
    '''
    
    def __init__(self, controls, cells, degree=3, points=None):
        
        self.controls = np.asarray(controls)
        self.cells = np.asarray(cells)
        self.ndim = self.controls.shape[1]
        self.degree = np.ones(self.ndim, dtype=int)*int(degree)

        # generate FFD lattice
        self.make_lattice()
        
        if not points is None: 
            self.set_points(points)
    
    
    
    def get_coords(self, points, use_clib=False, nitermax=10):
        coords = np.zeros((len(points),self.ndim), dtype=np.float64)
        icells = -np.ones((len(points)), dtype=np.int32)
        
        if self.ndim==2:
            ## 2d (pure python)
        
            for ic, cell in enumerate(self.cells):
                inside = self.is_inside_cell2d(points, cell)
                if inside.any():
                    icells[inside] = ic
                    coords[inside,:] = self.get_coords_cell2d(points[inside,:], cell)
                    
        elif self.ndim==3:
            ## 3d
        
            if use_clib:
                ## use ctypes lib (requires compilation)
                _pts = np.asarray(points, dtype=np.float64, order='C')
                _nodes = np.asarray(self.controls, dtype=np.float64, order='C')
                _cells = np.asarray(self.cells, dtype=np.uint32, order='C')
                libffd.get_coords_3d(
                    len(points), _pts, 
                    len(self.controls), _nodes,
                    len(self.cells), _cells,
                    nitermax,
                    coords, icells,
                    )
            else:
                ## pure python
                for ic, cell in enumerate(self.cells):
                    inside = self.is_inside_cell3d(points, cell)
                    if inside.any():
                        icells[inside] = ic
                        coords[inside,:] = self.get_coords_cell3d(
                            points[inside,:], 
                            cell, 
                            nitermax,
                            )
                
        return icells, coords
        
        
    def is_inside_cell2d(self, points, cell):
        tri1 = self.controls[[cell[0],cell[1],cell[2]]]
        tri2 = self.controls[[cell[1],cell[3],cell[2]]]

        inside = self.in_triangle(points,tri1) +\
                 self.in_triangle(points,tri2)
        return inside 
        
        
    def in_triangle(self, points, triangle):
        ''' test if 2d point P reside inside triangle ABC
            AP is between AB and AC
            BP is between BA and BC
            CP is between CA and CB
        '''
        AB = np.asarray(triangle[1] - triangle[0])
        BC = np.asarray(triangle[2] - triangle[1])
        CA = np.asarray(triangle[0] - triangle[2])
        
        inside = (0 <= np.multiply(
            np.cross(points - [triangle[0]], AB),
            np.cross(points - [triangle[0]], CA),
            )) & (0 <= np.multiply(
            np.cross(points - [triangle[1]], BC),
            np.cross(points - [triangle[1]], AB),
            )) & (0 <= np.multiply(
            np.cross(points - [triangle[2]], CA),
            np.cross(points - [triangle[2]], BC),
            ))
        # 1/0
        return inside
        
        
    def get_coords_cell2d(self, points, cell):
        ''' solve AP = s(1-t)AB + t(1-s)AC + st AD
        '''
        AP = points - [self.controls[cell[0]]]
        AB = self.controls[cell[1]] - self.controls[cell[0]]
        AC = self.controls[cell[2]] - self.controls[cell[0]]
        BD = self.controls[cell[3]] - self.controls[cell[1]]
        CD = self.controls[cell[3]] - self.controls[cell[2]]
        
        S = np.zeros(points.shape)
        
        ''' s1 '''
        a1 = np.cross(AB, AC-BD)
        b1 = - np.cross(AB, AC) - np.cross( AP, AC-BD)
        c1 = np.cross(AP, AC)
        
        if np.abs(a1)<1e-5:
            S[:,0] = -c1/b1
        else:
            delta1 = np.square(b1) - 4*a1*c1
            s11 = (-b1 - np.sqrt(delta1)) / (2.*a1)
            s12 = (-b1 + np.sqrt(delta1)) / (2.*a1)
            S[:,0] = s11*(0 <= s11)*(s11 <= 1) + s12*(0 <= s12)*(s12 <= 1)
        
        
        ''' s2 '''
        a2 = np.cross(AC, AB-CD)
        b2 = - np.cross(AC, AB) - np.cross( AP, AB-CD)
        c2 = np.cross(AP, AB)
        
        if np.abs(a2)<1e-5:
            S[:,1] = -c2/b2
        else:
            delta2 = np.square(b2) - 4*a2*c2
            s21 = (-b2 - np.sqrt(delta2)) / (2.*a2)
            s22 = (-b2 + np.sqrt(delta2)) / (2.*a2)
            S[:,1] = s21*(0 <= s21)*(s21 <= 1) + s22*(0 <= s22)*(s22 <= 1)
        
        return S
    

    def get_coords_cell3d(self, points, cell, nitermax=10, thresh=1e-3):
        ''' compute coordinates in cell
            
            A, B, C, ... are the vertices of the cell
            
            let : X = [s,t,u]^t
            and :
            F(s,t,u) = [A, B, C, D, E, F, G, H] x 
                [
                (1-s)(1-t)(1-u),
                s    (1-t)(1-u),
                (1-s)t    (1-u),
                s    t    (1-u),
                (1-s)(1-t)u,
                s    (1-t)u,
                (1-s)t    u,
                s    t    u,
                ] - Point
                
                J(s,t,u) is the Jacobian matrix of F(s,t,u)
                
            Newton approximation method:
                Xn+1 = Xn - inv(J(Xn)) F(Xn) 
                
        '''
        
        ndim = self.ndim
        if not len(points): return np.zeros((0,ndim))
        
        control = self.controls[cell]
        x = 0.5*np.ones(points.shape)
        
        for i in range(nitermax):
            print i
            c = np.c_[tuple([v1*v2*v3 \
                for v1 in [1-x[:,[0]], x[:,[0]]]\
                for v2 in [1-x[:,[1]], x[:,[1]]]\
                for v3 in [1-x[:,[2]], x[:,[2]]]\
                ])]
            
            F = np.dot(c, control) - points
            
            if np.max(F) < thresh: break
            
            J1 = np.dot(np.c_[tuple([v1*v2*v3 \
                for v1 in [-1., 1.]\
                for v2 in [1-x[:,[1]], x[:,[1]]]\
                for v3 in [1-x[:,[2]], x[:,[2]]]\
                ])], control)
                
            J2 = np.dot(np.c_[tuple([v1*v2*v3 \
                for v1 in [1-x[:,[0]], x[:,[0]]]\
                for v2 in [-1., 1.]\
                for v3 in [1-x[:,[2]], x[:,[2]]]\
                ])], control)
                
            J3 = np.dot(np.c_[tuple([v1*v2*v3 \
                for v1 in [1-x[:,[0]], x[:,[0]]]\
                for v2 in [1-x[:,[1]], x[:,[1]]]\
                for v3 in [-1., 1.]\
                ])], control)
                
            ## inverse = [ v2xv3 | v3xv1 | v1xv2 ]^t * 1/det
            invJ1 = np.cross(J2, J3)
            invJ2 = np.cross(J3, J1)
            invJ3 = np.cross(J1, J2)
            
            detJ = np.c_[np.sum(J1*invJ1, axis=1)]
            
            x = x - np.c_[
                np.sum(F*invJ1, axis=1),
                np.sum(F*invJ2, axis=1),
                np.sum(F*invJ3, axis=1),
                ]/detJ
            
        return x
        

        
        
    def is_inside_cell3d(self, points, cell):
        
        tetra1 = self.controls[[cell[0],cell[1],cell[2],cell[4]]]
        tetra2 = self.controls[[cell[1],cell[3],cell[2],cell[7]]]
        tetra3 = self.controls[[cell[4],cell[5],cell[7],cell[1]]]
        tetra4 = self.controls[[cell[4],cell[7],cell[6],cell[2]]]
        tetra5 = self.controls[[cell[1],cell[2],cell[4],cell[7]]]
        
        inside = self.in_tetrahedron(points,tetra1) +\
                 self.in_tetrahedron(points,tetra2) +\
                 self.in_tetrahedron(points,tetra3) +\
                 self.in_tetrahedron(points,tetra4) +\
                 self.in_tetrahedron(points,tetra5)
        return inside 
            
            
    def in_tetrahedron(self, points, tetrahedron):
        ''' test if points reside inside a tetrahedron
        Let the tetrahedron have vertices

        V1 = (x1, y1, z1)
        V2 = (x2, y2, z2)
        V3 = (x3, y3, z3)
        V4 = (x4, y4, z4)

        and your test point be P = (x, y, z).

    Then the point P is in the tetrahedron if following five determinants 
    all have the same sign.

             |x1 y1 z1 1|
        D0 = |x2 y2 z2 1|
             |x3 y3 z3 1|
             |x4 y4 z4 1|

             |x  y  z  1|
        D1 = |x2 y2 z2 1|
             |x3 y3 z3 1|
             |x4 y4 z4 1|

             |x1 y1 z1 1|
        D2 = |x  y  z  1|
             |x3 y3 z3 1|
             |x4 y4 z4 1|

             |x1 y1 z1 1|
        D3 = |x2 y2 z2 1|
             |x  y  z  1|
             |x4 y4 z4 1|

             |x1 y1 z1 1|
        D4 = |x2 y2 z2 1|
             |x3 y3 z3 1|
             |x  y  z  1|

        '''
        
        pts = np.array(points).reshape((-1,3))
        n_pt = pts.shape[0]
        tetras = np.hstack((np.array(tetrahedron),np.ones((4,1))))

        cofs = [[],[],[],[]]
        d0 = 0
        for i in range(4):
            indices = range(4)
            indices.remove(i)
            ## cofactors
            cofs[i] = [
                np.linalg.det(tetras[indices][:,[1,2,3]])*(-1.)**i,
                np.linalg.det(tetras[indices][:,[0,2,3]])*(-1.)**(i+1),
                np.linalg.det(tetras[indices][:,[0,1,3]])*(-1.)**(i),
                np.linalg.det(tetras[indices][:,[0,1,2]])*(-1.)**(i+1),
                ]
            d0 += tetras[0,i]*cofs[0][i]
        s0 = np.sign(d0)
        inside = np.repeat((True),n_pt)
        # S = []
        for i in range(4):
            s= pts[:,0]*cofs[i][0] + \
               pts[:,1]*cofs[i][1] + \
               pts[:,2]*cofs[i][2] + \
                    1.0*cofs[i][3]
            # S.append(float(s))
            inside *= (np.abs(s)<1e-5) + (np.sign(s)==s0)
        # print 'c' 
        # if inside.ravel()[0]:
            # print S
        
        return inside
    
        
#-------------------------------------------------------------------------------
if 1:
    ''' load library '''
    import os
    import platform
    import ctypes
    from numpy.ctypeslib import ndpointer
    
    if platform.architecture()[0]=='64bit': build_dir = 'build64'
    else: build_dir = 'build32'
    
    libpath = '%s/libs/libffdlib.so' %(build_dir)
    path = os.path.abspath(os.path.dirname(__file__))
    if not len(path):
        path = './'
    file = path + '//' + libpath
    print file
    if os.path.isfile(file):
        libffd = ctypes.CDLL(file)


        # arg types
        libffd.deform_3d.argtypes = [
            ctypes.c_uint32,  # npoint
            ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS'), # coords
            ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS'),    # icell
            ctypes.c_uint32,  # nnode
            ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS'),    # nodes
            ctypes.c_uint32,  # ncell
            ndpointer(dtype=np.uint32,ndim=2,flags='C_CONTIGUOUS'),    # cells
            ndpointer(dtype=np.uint32, shape=3),# degree
            ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS'), # results
            ]
    
        libffd.get_coords_3d.argtypes = [
            ctypes.c_uint32,  # npoint
            ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS'), # points
            ctypes.c_uint32,  # nnode
            ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS'), # nodes
            ctypes.c_uint32,  # ncell
            ndpointer(dtype=np.uint32,ndim=2,flags='C_CONTIGUOUS'),    # cells
            ctypes.c_uint32,  # nitermax
            ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS'), # coords
            ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS'),     # icells
            ]
    
#-------------------------------------------------------------------------------
if __name__=='__main__':
    import sys
    use_clib = False
    if '--use_clib' in sys.argv:
        i = sys.argv.index('--use_clib')
        use_clib = int(sys.argv[i+1])
        
    ndim = 3
    extent = [0,0,0,1,1,1]
    # points = np.asarray([[0,0,0], [0, 0.5, 0.5]])
    points = np.random.random((30, ndim))*extent[ndim:]
    points[:,0] = 0
    grid_shape = [2,2,2]
    
    ## init ffd
    # ffd = FFD(extent, shape=grid_shape)
    
    nodes = np.asarray([
        [0, 0, 0],
        [0, 0, 2],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 2],
        [1, 1, 0],
        [1, 1, 2],
        ])
    cells = np.asarray([[0,1,2,3,4,5,6,7]])
    edges = FFD([0,0,0,1,1,1], shape=(2,2,2)).edges
    
    ffd = EFFD(nodes, cells)
    controls = np.copy(ffd.controls).astype(float)
    # controls[-1] += [0,1,1]
    controls += np.random.random(controls.shape)*extent[ndim:]/grid_shape
    controls[:,0] = ffd.controls[:,0]
    
    ## compute new coordinates
    ffd.setpoints(points, use_clib=use_clib)
    dpoints = ffd.deform(controls, use_clib=use_clib)
    # 1/0
    
    effd = EFFD(controls, ffd.cells)
    effd.setpoints(dpoints, use_clib=use_clib)
    # 1/0
    
    dpoints2 = effd.deform(ffd.controls, use_clib=use_clib)
    
    
    
    # uncomment for plotting test
    from matplotlib import pyplot
    from rmn.plot import imgraph_3d
    fig = pyplot.figure(figsize=(10,6))
    
    pyplot.subplot(121)
    pyplot.axis('equal')
    fig1 = imgraph_3d.plot_graph(
        ffd.controls, 0, edges=edges, title='initial', figure=fig)
    imgraph_3d.plot_graph(points, 0, figure=fig1)
    
    pyplot.subplot(122)
    pyplot.axis('equal')
    fig2 = imgraph_3d.plot_graph(
        controls, 0, edges=edges, title='deformed', figure=fig)
    imgraph_3d.plot_graph(dpoints, 0, figure=fig2)
    
    
    
    
    
    
    
    
   
