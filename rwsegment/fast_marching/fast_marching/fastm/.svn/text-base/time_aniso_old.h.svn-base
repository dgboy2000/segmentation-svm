#ifndef __distance_aniso_old_h__
#define __distance_aniso_old_h__

#include <cmath>
#include <algorithm>

#include "common.h"


///-----------------------------------------------------------------------------
class ArrivalTime_aniso_old: public ArrivalTime
{
  public:
  
    /** compute arrival time */
    virtual float compute_time(unsigned int const i,
                               unsigned int const j,
                               unsigned int const k,
                               unsigned int const oi, // origin: not used
                               unsigned int const oj, // origin: not used
                               unsigned int const ok) // origin: not used
    {
        /// neighborhood type
        unsigned int nsize = neighborhood_size[_neigh_type];
        int (*neigh)[3] = (_neigh_type==1) ? neighborhood26 : neighborhood6;
    
        float *d = new float[nsize]; 
        float *s = new float[nsize];
        int *N = new int[nsize]; 
        float val = GET(_I,i,j,k);
        
        
        
        for (unsigned int u=0; u<nsize; u++) { d[u] = 1e10; N[u] = 0; s[u]=0;}
        
        for (unsigned int in=0; in<nsize; ++in)
        {
            int _i = neigh[in][0] + i;
            int _j = neigh[in][1] + j;
            int _k = neigh[in][2] + k;
            if ((_i>=0) && (_i<(int)ni) &&
                (_j>=0) && (_j<(int)nj) &&
                (_k>=0) && (_k<(int)nk) && GET(_Q,_i,_j,_k)>=0)
            {
                float w =(float)std::sqrt(1.*(i!=_i) + 1.*(j!=_j) + 1.*(k!=_k));
                float offset = _offset *w;
                    
                s[in] = std::abs(GET(_I,_i,_j,_k) - val) + offset;
                d[in] = GET(_D,_i,_j,_k);
                N[in] = 1;
            }   
        }
        
        float time = 1e10;
        for (unsigned u=0; u<nsize; u++)
            if (time > (d[u] + s[u]))
                time = (d[u] + s[u]);
        
        delete[] d;
        delete[] s;
        delete[] N;
        
        return time;
    }
    
    /** duration between two pixels */
    virtual float compute_interval(unsigned int const i1,
                                   unsigned int const j1,
                                   unsigned int const k1,
                                   unsigned int const i2,
                                   unsigned int const j2,
                                   unsigned int const k2)
    {
        float offset = _offset *
             (float)std::sqrt(1.*(i1!=i2) + 1.*(j1!=j2) + 1.*(k1!=k2));
        return std::abs( GET(_I, i1, j1, k1) - GET(_I, i2, j2, k2) ) + offset;
        
    }
    
    
    /** compute delta */
    virtual float compute_delta(unsigned int heap_size) const
    {
        /// neighborhood type
        unsigned int nsize = neighborhood_size[_neigh_type];
        int (*neigh)[3] = (_neigh_type==1) ? neighborhood26 : neighborhood6;
    
        /// compute minimum gap (gradient)
        float max_gap = 0;
        float g = 0;
        for (int i=0; i < (int)ni; i++)
        for (int j=0; j < (int)nj; j++)
        for (int k=0; k < (int)nk; k++)
        {            
            float v = GET(_I,i,j,k);
            for (unsigned int in=0; in<nsize/2; in++)
            {
                int _i = neigh[in][0] + i;
                int _j = neigh[in][1] + j;
                int _k = neigh[in][2] + k;
                if ((_i>=0) && (_i<(int)ni) &&
                    (_j>=0) && (_j<(int)nj) &&
                    (_k>=0) && (_k<(int)nk))
                {
                    float w =(float)std::sqrt(1.*(i!=_i) + 1.*(j!=_j) + 1.*(k!=_k));
                    float offset = _offset *w;
                    g = std::abs(v - GET(_I, _i, _j, _k)) + offset;
                    if (g > max_gap) max_gap = g;
                }
            }
            
        }
        ///+1 for precision errors
        float delta = (max_gap+1)/((float)heap_size); 
        return delta;
    }

    /** constructor */
    ArrivalTime_aniso_old(unsigned int const size[3], float offset,
              float const *I, int const *Q, float const *D, 
              int neighborhood_type=0)
    : ArrivalTime(size, offset, I, Q, D, neighborhood_type) 
    { std::cout << "Using Arrival Time Anisotropic method\n";}
    
};

#endif
