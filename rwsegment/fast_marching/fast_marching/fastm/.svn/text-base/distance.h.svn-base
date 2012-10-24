#ifndef __distance_h__
#define __distance_h__

#include <cmath>
#include <algorithm>


///-----------------------------------------------------------------------------
class Distance
{
  public:
  
    float operator()(unsigned int i, 
                     unsigned int j, 
                     unsigned int k,
                     float offset) const
    {
        
        float *d = new float[_nsize]; 
        float *N = new float[_nsize];
        
        
        for (unsigned int u=0; u<_nsize; u++) { d[u] = 1e10; N[u] = 0;}
        
        
        for (unsigned int in=0; in<_nsize; ++in)
        {
            int _i = _neigh[in][0] + i;
            int _j = _neigh[in][1] + j;
            int _k = _neigh[in][2] + k;
            if ((_i>=0) && (_i<(int)ni) &&
                (_j>=0) && (_j<(int)nj) &&
                (_k>=0) && (_k<(int)nk) && GET(_Q,_i,_j,_k)>=0)
            {
                float w = 1.*(i!=_i) + 1.*(j!=_j) + 1.*(k!=_k);
                // float _offset = offset *w;
                d[in] = GET(_D,_i,_j,_k);
                N[in] = 1./w;
            }   
        }
        
        /// speed
        float speed = GET(_I,i,j,k) + offset;
        float res2 = 1/(speed*speed);

        
        float b=0, d2=0, N_=0;
        for (unsigned int u=0; u!=_nsize; u++)
        {
            b += d[u]*N[u];
            d2 += d[u]*d[u]*N[u];
            N_ += N[u];
        }
        
        float Delta;
        Delta = b*b - N_*(d2 - res2);
        
        float s;
        if (Delta < 0)
        {
            /// if speed too high, use only closest neighbor
            s = *std::min_element(d,d+_nsize) + 1/speed;
        }
        else
        {
            /// standard distance
            s = (b + std::sqrt(Delta)) / N_;
        }
        
        // if (s!=s) std::cout 
        // if (i==45 && j==70 && k==110) std::cout 
            // << "Nan! at" << i << " " << j << " " << k << "\n"
            // << "image size = " << ni << " " << nj << " " << nk << "\n"
            // << "d = " << d[0] << " " << d[1] << " " << d[2] << " "
                      // << d[3] << " " << d[4] << " " << d[5] << "\n"
            // << "N = " << N[0] << " " << N[1] << " " << N[2] << " "
                      // << N[3] << " " << N[4] << " " << N[5] << "\n"
            // << "b = " << b << ", N_ = " << N_ << "\n"
            // << "Delta = " << Delta << ", speed = " << speed << "\n"
            // << "dist = " << s << "\n"
            // ;
        
        delete[] d;
        delete[] N;
        
        return s;
    }

    
    Distance( unsigned int const size[3],
              float const *I, int const *Q, float const *D, 
              int const (*neigh)[3], unsigned int neigh_size)
              
    : _I(I), _Q(Q), _D(D), 
      _neigh(neigh), _nsize(neigh_size),
      ni(size[0]), nj(size[1]), nk(size[2]) {}
    
  private:
    int const *_Q;
    float const *_D;
    float const *_I;
    int const (*_neigh)[3];
    unsigned int _nsize;    
    unsigned int ni,nj,nk;
    // bool _fastxyz;
};
///-----------------------------------------------------------------------------
#endif
