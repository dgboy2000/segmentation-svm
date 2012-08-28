#include <iostream>
#include <cmath>
#include <list>


#define GETID(_i,_j,_k) ((_i)*Nk*Nj + (_j)*Nk + (_k))
#define GET(_img,_i,_j,_k) (_img[(_i)*Nk*Nj + (_j)*Nk + (_k)])
#define SET(_img,_i,_j,_k,v) (_img[(_i)*Nk*Nj + (_j)*Nk + (_k)] = (v))

#define ISIN(_i,_j,_k) ((_i>=0)&&(_j>=0)&&(_k>=0)&&(_i<(int)Ni)&&(_j<(int)Nj)&&(_k<(int)Nk))

/** allocator
 */
typedef void*(*Allocator)(int, unsigned int*, char*, char*);


///-----------------------------------------------------------------------------
///-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  extern __declspec(dllexport) 
#elif _WIN64
  extern __declspec(dllexport)
#endif 
///-----------------------------------------------------------------------------
/** patch difference
 */
int patch_diff_3d(
    unsigned int shape[3],
    float const *image,
    unsigned int radius[2],
    unsigned int step,
    Allocator allocator,
    bool gaussian_weighting
    )
{
    unsigned int Ni = shape[0], Nj = shape[1], Nk = shape[2];
    unsigned int R0 = radius[0], R1 = radius[1];
    
    typedef std::list<std::pair<unsigned int, unsigned int> > IndiceType;
    IndiceType indices;
    std::list<float> values;
    
    float sigma2 = (float)(1/9.0)*(R0+1)*(R0+1);
    
    unsigned int dim[3];
    for (unsigned int d=0; d<3; d++)
    {
        dim[0] = (d==0?1:0);
        dim[1] = (d==1?1:0);
        dim[2] = (d==2?1:0);
    
        for (unsigned int ni=0; ni<Ni-dim[0]; ni++)
        for (unsigned int nj=0; nj<Nj-dim[1]; nj++)
        for (unsigned int nk=0; nk<Nk-dim[2]; nk++)
        {
            /// patch centers
            std::pair<unsigned int, unsigned int> pair;
            pair.first = GETID(ni,nj,nk);
            pair.second = GETID(ni+dim[0],nj+dim[1],nk+dim[2]);
            
            float v = 0;
            float nv = 0;
            // std::cout << "pair " << pair.first << " " << pair.second << "\n";
            
            /// loop through patches
            int Rbegin[3];
            int Rend[3];
            
            Rbegin[0] = -(dim[0]==1?(int)R0:(int)R1);
            Rbegin[1] = -(dim[1]==1?(int)R0:(int)R1);
            Rbegin[2] = -(dim[2]==1?(int)R0:(int)R1);
            Rend[0] =    (dim[0]==1?(int)R0 - step + 1:(int)R1);
            Rend[1] =    (dim[1]==1?(int)R0 - step + 1:(int)R1);
            Rend[2] =    (dim[2]==1?(int)R0 - step + 1:(int)R1);
            
            for (int ri=Rbegin[0]; ri<=Rend[0]; ri++)
            for (int rj=Rbegin[1]; rj<=Rend[1]; rj++)
            for (int rk=Rbegin[2]; rk<=Rend[2]; rk++)
            {
                int i1 = (int)ni + (int)ri;
                int j1 = (int)nj + (int)rj;
                int k1 = (int)nk + (int)rk;
                
                int i2 = i1 + (int)(dim[0]==1?step:0);
                int j2 = j1 + (int)(dim[1]==1?step:0);
                int k2 = k1 + (int)(dim[2]==1?step:0);
                
                //std::cout 
                //    << "\tpos" << i1 << " " << j1 << " " << k1 << "\n"
                //    << "\tpos" << i2 << " " << j2 << " " << k2 << "\n"
                //    ;
                
                if (ISIN(i1,j1,k1)&&ISIN(i2,j2,k2))
                {
                    float v1 = GET(image, i1,j1,k1);
                    float v2 = GET(image, i2,j2,k2);
                    v += (v1 - v2)*(v1 - v2);
                    if (gaussian_weighting)
                        nv += (float)std::exp(- 1/sigma2 * (
                            (0.5*i1 + 0.5*i2 - ni)*(0.5*i1 + 0.5*i2 - ni) +
                            (0.5*j1 + 0.5*j2 - nj)*(0.5*j1 + 0.5*j2 - nj) +
                            (0.5*k1 + 0.5*k2 - nk)*(0.5*k1 + 0.5*k2 - nk))
                            );
                    else
                        nv += 1;
                    
                    //std::cout 
                    //    << "\tpos" << GETID(i1,j1,k1) << ": " << v1 << "\n"
                    //    << "\tpos" << GETID(i2,j2,k2) << ": " << v2 << "\n"
                    //    ;
                }
            }
            
            /// save pair
            indices.push_back(pair);
            values.push_back(v/nv);
        }
    }
    
    // allocate output
    unsigned int lshape[] = {(unsigned int)indices.size(),2};
    typedef long long int int64; 
    typedef double float64;
    int64 *inds = (int64*) allocator(2, lshape, "int64", "ij");
    unsigned int vshape[] = {lshape[0]};
    float64 *vals = (float64*) allocator(1,vshape, "float64", "data");
    
    IndiceType::const_iterator iindices = indices.begin();
    std::list<float>::const_iterator ivalues = values.begin();
    int64 *iinds = inds;
    float64 *ivals = vals;
    for (; iindices!=indices.end(); ++iindices, ++ivalues, ++iinds, ++ivals)
    {
        (*iinds) = iindices->first;
        iinds ++;
        (*iinds) = iindices->second;
        (*ivals) = (*ivalues);
    }   
    
    return 0;
    
} /// end patch_diff()
///-----------------------------------------------------------------------------





///-----------------------------------------------------------------------------
///-----------------------------------------------------------------------------
#ifdef __cplusplus
}
#endif
