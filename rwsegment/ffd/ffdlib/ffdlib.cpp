#include <iostream>
#include <cmath>

int factorial (int num)
{
 if (num<=1)
    return 1;
 return factorial(num-1)*num; // recursive call
}

int comb(int n, int k)
{
    return factorial(n)/(factorial(k)*factorial(n-k));
}

bool in_tetrahedron(double const [3], double const [4][3], bool);
void compute_3d_coordinates(
    double const [3], double const [8][3], double [3], unsigned int, double);
double determinant4(double const [4][4]);
    
///-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif


/** Move points according to the position of the control points
    P = sum(c) sum(i,j,k) CI^i CJ^j CK^k s^i t^j u^k (1-s)^(I-i) (1-t)^(J-j) (1 - u)^(K-k) Pcijk
 */
//extern __declspec(dllexport) 
int deform_3d(
    unsigned int npoint, 
    double const *coords,
    int const *icells,
    unsigned int nnode,
    double const * nodes,
    unsigned int ncell,
    unsigned int const *cells,
    unsigned int degree[3],
    double * results
    )
    
{
    unsigned int L = degree[0];
    unsigned int M = degree[1];
    unsigned int N = degree[2];

    int S = (L+1)*(M+1)*(N+1);
    
    /// allocate intermediary nodes P_c,i,j,k
    double * P = new double[3*ncell*S];
    
    for (unsigned int c=0; c<ncell; c++)
    for (unsigned int l=0; l<=L; l++)
    for (unsigned int m=0; m<=M; m++)
    for (unsigned int n=0; n<=N; n++)
    {
        
        unsigned int iP = l*(M+1)*(N+1) + m*(N+1) + n;
        double s = double(l)/L;
        double t = double(m)/M;
        double u = double(n)/N;
        for (unsigned int d=0; d<3; ++d)
            P[c*S*3 + iP*3 + d] = 
                nodes[cells[c*8 + 0]*3 + d]*(1-s)*(1-t)*(1-u)+
                nodes[cells[c*8 + 1]*3 + d]*(1-s)*(1-t)*   u +
                nodes[cells[c*8 + 2]*3 + d]*(1-s)*   t *(1-u)+
                nodes[cells[c*8 + 3]*3 + d]*(1-s)*   t *   u +
                nodes[cells[c*8 + 4]*3 + d]*   s *(1-t)*(1-u)+
                nodes[cells[c*8 + 5]*3 + d]*   s *(1-t)*   u +
                nodes[cells[c*8 + 6]*3 + d]*   s *   t *(1-u)+
                nodes[cells[c*8 + 7]*3 + d]*   s *   t *   u;
    }
    
    /// interpolate points
    for (unsigned int i=0; i<npoint; i++)
    {
        
        int c = icells[i]; // cell index
        
        if ((c < 0)||(c >= (int)ncell))
            continue;
            
        results[i*3 + 0] = 0;
        results[i*3 + 1] = 0;
        results[i*3 + 2] = 0;
        
        double s = coords[i*3 + 0];
        double t = coords[i*3 + 1];
        double u = coords[i*3 + 2];
        
        
        for (unsigned int l=0; l<=L; l++)
        for (unsigned int m=0; m<=M; m++)
        for (unsigned int n=0; n<=N; n++)
        {
        
            unsigned int iP = l*(M+1)*(N+1) + m*(N+1) + n;
            
            double pows = 
                (l==0?1.:std::pow(s, (int)l)) *
                (l==L?1.:std::pow(1-s, (int)(L-l))) *
                (m==0?1.:std::pow(t, (int)m)) *
                (m==M?1.:std::pow(1-t, (int)(M-m))) *
                (n==0?1.:std::pow(u, (int)n)) *
                (n==N?1.:std::pow(1-u, (int)(N-n)));
            
            double combs = comb(L,l)*comb(M,m)*comb(N,n);
            
            results[i*3 + 0] += P[c*S*3 + iP*3 + 0]*pows*combs;
            results[i*3 + 1] += P[c*S*3 + iP*3 + 1]*pows*combs;
            results[i*3 + 2] += P[c*S*3 + iP*3 + 2]*pows*combs;
        }
    }
    
    delete[] P;
    
    
    return 0;
    
} /// defrom_3d()
///-----------------------------------------------------------------------------


///-----------------------------------------------------------------------------
//extern __declspec(dllexport) 
int get_coords_3d(
    unsigned int npoint, 
    double const *points,
    unsigned int nnode,
    double const * nodes,
    unsigned int ncell,
    unsigned int const *cells,
    unsigned int nitermax,
    double * coords,
    int * icells
    )
{
    /// consts
    const int ndim = 3;
    const int nvertex = 8;
    const int ntetra = 5;
    const int ntetravertex = 4;
    
    double thresh = 1e-5;
    
    /// tetrahedra    
    const int tetras[5][4] = {
        {0,1,2,4},
        {1,3,2,7},
        {4,5,7,1},
        {4,7,6,2},
        {1,2,4,7}};
        
    
    double vertices[nvertex][ndim];
    double tvertices[ntetravertex][ndim];
        
    /// initialize cell indices
    for (int * icell=icells; icell<icells+npoint; ++icell)
        icell[0] = -1;
    
    /// find cell index and compute coordinates
    for (unsigned int c=0; c < ncell; c++)
    {
    
        // std::cout << "cell = " << c << "\n";
    
        /// get nodes for current cell
        for (unsigned int v=0; v<nvertex; v++)
        for (unsigned int d=0; d<ndim; d++)
            vertices[v][d] = nodes[cells[c*nvertex + v]*ndim + d];
            
        
        for (unsigned int t=0; t<ntetra; t++)
        {
            // std::cout << "\ttetra = " << t << "\n";
        
            /// get nodes for current tetrahedron
            for (unsigned int v=0; v<ntetravertex; v++)
            for (unsigned int d=0; d<ndim; d++)
            {
                tvertices[v][d] = vertices[tetras[t][v]][d];
            }
            
            /// find which points are inside the tetrahedron
            double const *p=points;
            int *icell = icells;
            double * coord = coords;
            for (; p<(points+npoint*3); p+=3, ++icell, coord+=3)
            {
                // bool f = 0;
                // if ((c==9)||(c==12))
                // {
                    // f=1;
                    // std::cout
                    // << "in tetra ? " << in_tetrahedron(p, tvertices,f) << "\n"
                    // ;
                // }
            
                /// if already set
                if (icell[0]>-1)
                    continue;
                    
                /// if not set and inside tetrahedron
                else if (in_tetrahedron(p, tvertices,0))
                {
                    icell[0] = c;
                    
                    /// compute coordinates in 3d cell
                    compute_3d_coordinates(
                        p, vertices, coord, nitermax, thresh);
                }
                
            } /// end for loop on points
        
        } /// end for loop on tetrahedra
    
    } /// end for loop on cells
    
    
    return 1;
} /// get_coords_3d()
///-----------------------------------------------------------------------------
    
#ifdef __cplusplus
}
#endif
    
///-----------------------------------------------------------------------------
void compute_3d_coordinates(
    double const point[3], 
    double const vertices[8][3], 
    double coord[3], unsigned int nitermax, double thresh=1e-5)
{
    double X[3] = {0.5, 0.5, 0.5};
    double F[3];
    double J[3][3];
    double invJ[3];

    
    for (unsigned int n=0; n<nitermax; n++)
    {
    
        F[0] = -point[0]; 
        F[1] = -point[1]; 
        F[2] = -point[2];
        double detJ = 0;
        double invJF[3] = {0,0,0};
        
        for (unsigned int j=0; j<9; j++)
            J[j/3][j%3] = 0;
            
        for (unsigned int v=0; v<8; v++)
        {
            double mult1 = ((v/4)>0   ? X[0] : 1-X[0]);
            double mult2 = ((v/2)%2>0 ? X[1] : 1-X[1]);
            double mult3 = ((v%2)>0   ? X[2] : 1-X[2]);
                
            for (unsigned int d=0; d<3; d++)
            {
                F[d] += mult1*mult2*mult3*vertices[v][d];
                J[0][d] += ((v/4)>0   ? 1 : -1)*mult2*mult3*vertices[v][d];
                J[1][d] += ((v/2)%2>0 ? 1 : -1)*mult1*mult3*vertices[v][d];
                J[2][d] += ((v%2)>0   ? 1 : -1)*mult1*mult2*vertices[v][d];

            }
        } /// end fill F and J
        
        if ((std::abs(F[0])<thresh)&&
            (std::abs(F[1])<thresh)&&
            (std::abs(F[2])<thresh))
            break;
        
        /// jacobian inverse irow1 = col2xcol3 / detJ
        for (unsigned int c=0; c<3; c++)
        {
            unsigned int c1 = (c+1)%3;
            unsigned int c2 = (c+2)%3;
            
            for (unsigned int r=0; r<3; r++)
            {
                unsigned int r1 = (r+1)%3;
                unsigned int r2 = (r+2)%3;
                
                invJ[r] = J[c1][r1]*J[c2][r2] - J[c1][r2]*J[c2][r1];
                invJF[c] += invJ[r]*F[r];
            }
            detJ += J[c][0]*invJ[0];
        }

        
        X[0] = X[0] - invJF[0]/detJ;
        X[1] = X[1] - invJF[1]/detJ;
        X[2] = X[2] - invJF[2]/detJ;
        
    }
    
    coord[0] = X[0];
    coord[1] = X[1];
    coord[2] = X[2];
    
} /// compute_3d_coordinates
///-----------------------------------------------------------------------------

///-----------------------------------------------------------------------------
bool in_tetrahedron(double const point[3], double const vertices[4][3], bool f)
{
    /** test if points reside inside a tetrahedron
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

        **/
        
    double matrix[4][4];
    matrix[0][3] = 1; matrix[1][3] = 1; matrix[2][3] = 1; matrix[3][3] = 1;
    
    for (unsigned int v=0; v<4; v++)
    for (unsigned int d=0; d<3; d++)
        matrix[v][d] = vertices[v][d];
    
    // if (f) std::cout 
        // << matrix[0][0] << " " << matrix[0][1] << " " 
        // << matrix[0][2] << " " << matrix[0][3] << "\n"
        // << matrix[1][0] << " " << matrix[1][1] << " " 
        // << matrix[1][2] << " " << matrix[1][3] << "\n"
        // << matrix[2][0] << " " << matrix[2][1] << " " 
        // << matrix[2][2] << " " << matrix[2][3] << "\n"
        // << matrix[3][0] << " " << matrix[3][1] << " " 
        // << matrix[3][2] << " " << matrix[3][3] << "\n"
        // ;
    
    
    /// D0
    double D0 = determinant4(matrix);
    bool sign = D0>0;
    
    // if (f) std::cout << "D0 " << D0 << "\n";
    
    /// D1
    for (unsigned int d=0; d<3; d++)
        matrix[0][d] = point[d];
    double D1 = determinant4(matrix);

    // if (f) std::cout << "D1 " << D1 << "\n";
    
    if ((std::abs(D1)>1e-5)&&((D1>0)!=sign))
        return false;
    
    /// D2
    for (unsigned int d=0; d<3; d++)
    {
        matrix[0][d] = vertices[0][d];
        matrix[1][d] = point[d];
    }
    double D2 = determinant4(matrix);
    
    // if (f) std::cout << "D2 " << D2 << "\n";
    
    if ((std::abs(D2)>1e-5)&&((D2>0)!=sign))
        return false;
        
    /// D3
    for (unsigned int d=0; d<3; d++)
    {
        matrix[1][d] = vertices[1][d];
        matrix[2][d] = point[d];
    }
    double D3 = determinant4(matrix);
    
    // if (f) std::cout << "D3 " << D3 << "\n";
    
    if ((std::abs(D3)>1e-5)&&((D3>0)!=sign))
        return false;
        
    /// D4
    for (unsigned int d=0; d<3; d++)
    {
        matrix[2][d] = vertices[2][d];
        matrix[3][d] = point[d];
    }
    double D4 = determinant4(matrix);
    
    // if (f) std::cout << "D4 " << D4 << "\n";
    
    if ((std::abs(D4)>1e-5)&&((D4>0)!=sign))
        return false;
      
    return true;
    
} /// in_tetrahedron
///-----------------------------------------------------------------------------

double determinant4(double const M[4][4])
{
    double det = 0;
    for (unsigned int i=0; i<4; i++)
    {
        int ia = (i>0?0:1);
        int ib = (i>1?1:2);
        int ic = (i>2?2:3);
        
        double cof = 
            M[ia][1] * (M[ib][2]*M[ic][3] - M[ic][2]*M[ib][3]) +
            M[ib][1] * (M[ic][2]*M[ia][3] - M[ia][2]*M[ic][3]) +
            M[ic][1] * (M[ia][2]*M[ib][3] - M[ib][2]*M[ia][3]);
            
        det += M[i][0] * cof * ((i%2)==0?1:-1);
    }
    return det;
}


