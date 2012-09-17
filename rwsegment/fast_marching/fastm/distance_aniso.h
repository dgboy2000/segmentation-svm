#ifndef __time_aniso_h__
#define __time_aniso_h__

#include <cmath>
#include <algorithm>


///-----------------------------------------------------------------------------
class DistanceAniso
{
  public:
    float operator()(unsigned int i, 
                     unsigned int j, 
                     unsigned int k,
                     float offset) const
    {
        
        float *d = new float[_nsize]; 
        float *s = new float[_nsize];
        int *N = new int[_nsize]; 
        float val = GET(_I,i,j,k);
        
        for (unsigned int u=0; u<_nsize; u++) { d[u] = 1e10; N[u] = 0; s[u]=0;}
        
        for (unsigned int in=0; in<_nsize; ++in)
        {
            int _i = _neigh[in][0] + i;
            int _j = _neigh[in][1] + j;
            int _k = _neigh[in][2] + k;
            if ((_i>=0) && (_i<(int)ni) &&
                (_j>=0) && (_j<(int)nj) &&
                (_k>=0) && (_k<(int)nk) && GET(_Q,_i,_j,_k)>=0)
            {
                float w =(float)std::sqrt(1.*(i!=_i) + 1.*(j!=_j) + 1.*(k!=_k));
                float _offset = offset *w;
                    
                s[in] = std::abs(GET(_I,_i,_j,_k) - val) + _offset;
                // s[in] = std::abs(GET(_I,_i,_j,_k) - val)*w + _offset;
                // d[in] = GET(_D,_i,_j,_k) + s[in];
                d[in] = GET(_D,_i,_j,_k);
                N[in] = 1;
            }   
        }
        
        float dist;
        
        /// method 1
        if (_method==1)
        {
            float S = 0;
            for (unsigned int u=0; u<_nsize; u++)
                S += N[u]/(s[u]+(float)1e-10);
                
            dist = 0;
            for (unsigned int u=0; u<_nsize; u++)
                dist += N[u]*(d[u]+s[u])/(s[u]+(float)1e-10)/S;
                
        }
        
        if (_method==2)
        {
            float speed = 0;
            float b=0, d2=0, N_=0;
            for (unsigned u=0; u<_nsize; u++)
            {
                speed += s[u]*s[u];
                b += d[u]*N[u];
                d2 += (d[u]*d[u])*N[u];
                N_ += N[u];
            }
            
            float res2 = 1/(speed);
            speed = (float)std::sqrt(speed);
            
            
            float Delta = b*b - N_*(d2 - res2);
            if (Delta < 0)
            {
                /// if speed too high, use only closest neighbor
                dist = *std::min_element(d,d+_nsize) + 1/speed;
            }
            else
            {
                /// standard distance
                dist = (b + std::sqrt(Delta)) / N_;
            }
            
        }
        if (_method==3)
        {
            dist = 1e10;
            for (unsigned u=0; u<_nsize; u++)
                if (dist > d[u])
                    dist = d[u];
                    
            float speed = 0;
            for (unsigned u=0; u<_nsize; u++)
                speed += s[u]*s[u];
                
            dist += std::sqrt(speed);
        }
        
        
        ///method 0
        else 
        {
            // dist = *std::min_element(d, d+_nsize);
            dist = 1e10;
            for (unsigned u=0; u<_nsize; u++)
                if (dist > (d[u] + s[u]))
                    dist = (d[u] + s[u]);
        }

        
        delete[] d;
        delete[] s;
        delete[] N;
        
        return dist;
    }
    
    float dist(unsigned int i1, unsigned int j1, unsigned int k1, 
               unsigned int i2, unsigned int j2, unsigned int k2, 
               float offset)
    {
        float _offset = offset *
             (float)std::sqrt(1.*(i1!=i2) + 1.*(j1!=j2) + 1.*(k1!=k2));
        return std::abs( GET(_I, i1, j1, k1) - GET(_I, i2, j2, k2) ) + _offset;
        
    }
    
    DistanceAniso( unsigned int const size[3],
              float const *I, int const *Q, float const *D, 
              int const (*neigh)[3], unsigned int neigh_size, int method=0)
              
    : _I(I), _Q(Q), _D(D), 
      _neigh(neigh), _nsize(neigh_size),
      ni(size[0]), nj(size[1]), nk(size[2]), _method(method) {}
    
  private:
    int const *_Q;
    float const *_D;
    float const *_I;
    int const (*_neigh)[3];
    unsigned int _nsize;
    unsigned int ni,nj,nk;
    int _method;
};

#endif
