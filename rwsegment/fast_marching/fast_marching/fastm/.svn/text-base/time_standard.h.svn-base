#ifndef __time_standard_h__
#define __time_standard_h__

#include <cmath>
#include <algorithm>

#include "common.h"


///-----------------------------------------------------------------------------
class ArrivalTime_standard: public ArrivalTime
{
  public:
  
    /** compute arrival time */
    virtual float compute_time(unsigned int i,
                               unsigned int j,
                               unsigned int k,
                               unsigned int oi, // origin: not used
                               unsigned int oj,
                               unsigned int ok) const
    {
        /// neighborhood type
        unsigned int nsize = neighborhood_size[_neigh_type];
        int (*neigh)[3] = (_neigh_type==1) ? neighborhood26 : neighborhood6;
    
        
        float *d = new float[nsize];
        float *N = new float[nsize];
        
        for (unsigned int u=0; u<nsize; u++) { d[u] = 1e10; N[u] = 0;}
        
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
                d[in] = GET(_D,_i,_j,_k);
                N[in] = 1/w;
            }   
        }
        
        float speed = GET(_I,i,j,k) + _offset;
        float res2 = 1/(speed*speed);
        
        float b=0, d2=0, N_=0;
        for (unsigned int u=0; u!=nsize; u++)
        {
            b += d[u]*N[u];
            d2 += d[u]*d[u]*N[u];
            N_ += N[u];
        }
        
        float Delta;
        Delta = b*b - N_*(d2 - res2);
        
        float time;
        if (Delta < 0)
        {
            /// if speed too high, use only closest neighbor
            time = *std::min_element(d,d+nsize) + 1/speed;
        }
        else
        {
            /// standard distance
            time = (b + std::sqrt(Delta)) / N_;
        }
        
        delete[] d;
        delete[] N;
        
        return time;
    }
    
    /** duration between two pixels */
    virtual float compute_interval(unsigned int i1,
                                   unsigned int j1,
                                   unsigned int k1,
                                   unsigned int i2,
                                   unsigned int j2,
                                   unsigned int k2) const
    { return 0.0; }
    
    
    /** compute delta */
    virtual float compute_delta(unsigned int heap_size) const
    {
        /// compute minimum gap (gradient)
        float minspeed = 1e10;
        for (int i=0; i < (int)ni; i++)
        for (int j=0; j < (int)nj; j++)
        for (int k=0; k < (int)nk; k++)
        {            
            float v = GET(_I,i,j,k);
            if (v < minspeed) minspeed = v;
        }
        float delta = 1/((minspeed+_offset)*heap_size);
        return delta;
    }
    

    /** constructor */
    ArrivalTime_standard(unsigned int const size[3], float offset,
              float const *I, int const *Q, float const *D, 
              int neighborhood_type=0)
    : ArrivalTime(size, offset, I, Q, D, neighborhood_type)
    { std::cout << "Using standard Arrival Time method\n";}
    
};

#endif
