#ifndef __distance_aniso_h__
#define __distance_aniso_h__

#include <cmath>
#include <algorithm>

#include "common.h"


///-----------------------------------------------------------------------------
class ArrivalTime_aniso: public ArrivalTime
{
  public:
  
    /** compute arrival time */
    virtual float compute_time(unsigned int i,
                               unsigned int j,
                               unsigned int k,
                               unsigned int oi,
                               unsigned int oj,
                               unsigned int ok) const
    {
        
        return this->compute_interval(i,j,k,oi,oj,ok) + GET(_D, oi, oj, ok);
    }
    
    /** duration between two pixels */
    virtual float compute_interval(unsigned int i1,
                                   unsigned int j1,
                                   unsigned int k1,
                                   unsigned int i2,
                                   unsigned int j2,
                                   unsigned int k2) const
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
                    (_k>=0) && (_k<(int)nk) && GET(_Q,_i,_j,_k)>=0)
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
    ArrivalTime_aniso(unsigned int const size[3], float offset,
              float const *I, int const *Q, float const *D, 
              int neighborhood_type=0)
    : ArrivalTime(size, offset, I, Q, D, neighborhood_type)
    { std::cout << "Using Arrival Time Anisotropic method\n";}
    
};

#endif
