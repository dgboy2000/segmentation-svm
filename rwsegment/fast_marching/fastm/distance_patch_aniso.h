#ifndef __distance_patch_aniso_h__
#define __distance_aniso_h__

#include <cmath>
#include <algorithm>

///-----------------------------------------------------------------------------
class DistancePatchAniso
{
  public:
  
    /** get patch values
     *
     */
    inline void get_patch(int i, int j, int k, float * patch) const
    {
        unsigned int u = 0;
        for (int _i=-(int)_radius; _i<=(int)_radius; ++_i)
        for (int _j=-(int)_radius; _j<=(int)_radius; ++_j)
        for (int _k=-(int)_radius; _k<=(int)_radius; ++_k)
        {
            if ( ((i+_i)<0) || ((i+_i)>=(int)ni) || 
                 ((j+_j)<0) || ((j+_j)>=(int)nj) || 
                 ((k+_k)<0) || ((k+_k)>=(int)nk) )
                patch[u] = GET(_I,i,j,k);
            else
                patch[u] = 0;
            u++;
        }
    }
   
    /// difference of means
    inline float dist_mean(
        float const *patch1, float const *patch2) const
    {
        float mean1 = 0, mean2 = 0;
        unsigned int diam = (2*_radius+1);
        unsigned int diam3 = diam*diam*diam;
        for (unsigned int u=0; u < diam3; ++u)
        {
            mean1 += patch1[u];
            mean2 += patch2[u];
        }
        return std::abs(mean1/diam3 - mean2/diam3);
    }
    
    /// histogram distance
    inline float dist_histo(
        float const *patch1, float const *patch2) const
    {
        /// to do
        return 0;
    }
    
    
    
  
    float operator()(unsigned int i, 
                     unsigned int j, 
                     unsigned int k, 
                     float offset) const
    {
        
        float *d = new float[_nsize]; 
        float *s = new float[_nsize];
        int *N = new int[_nsize]; 
        
        for (unsigned int u=0; u<_nsize; u++) { d[u] = 1e10; N[u] = 0; s[u]=0;}
        
        unsigned int diam = (2*_radius+1);
        unsigned int diam3 = diam*diam*diam;
        float *patch0 = new float[diam3];
        this->get_patch(i,j,k,patch0);
        
        float *patch1 = new float[diam3];
        float (DistancePatchAniso::*dist_func)
            (float const *, float const *) const = NULL;
        dist_func = &DistancePatchAniso::dist_mean;
        
        for (unsigned int in=0; in<_nsize; ++in)
        {
            int _i = _neigh[in][0] + i;
            int _j = _neigh[in][1] + j;
            int _k = _neigh[in][2] + k;
            if ((_i>=0) && (_i<(int)ni) &&
                (_j>=0) && (_j<(int)nj) &&
                (_k>=0) && (_k<(int)nk) && GET(_Q,_i,_j,_k)>=0)
            {
                float _offset = offset *
                    (float)std::sqrt(1.*(i!=_i) + 1.*(j!=_j) + 1.*(k!=_k));
            
                this->get_patch(i-1, j, k, patch1);
                s[in] = (*this.*dist_func)(patch0, patch1) + _offset;
                d[in] = GET(_D,_i,_j,_k) + s[in];
                N[in] = 1;
            }   
        }
        
        /*
        /// i-1
        if ((i>0) && (GET(_Q,i-1,j,k)>=0))
        {
            this->get_patch(i-1, j, k, patch1);
            s[0] = (*this.*dist_func)(patch0, patch1);
            d[0] = GET(_D,i-1,j,k) + s[0];
            N[0] = 1;
        }
        
        ///i+1
        if ((i<ni-1) && (GET(_Q,i+1,j,k)>=0))
        {
            // s[1] = std::abs(GET(_I,i+1,j,k) - val);
            this->get_patch(i+1, j, k, patch1);
            s[1] = (*this.*dist_func)(patch0, patch1);
            d[1] = GET(_D,i+1,j,k) + s[1];
            N[1] = 1;
            // di = std::min(
                // di,GET(_D,i+1,j,k) + std::abs(GET(_I,i+1,j,k) - val));
        }
        
        /// j-1
        if ((j>0) && (GET(_Q,i,j-1,k)>=0))
        {
            // s[2] = std::abs(GET(_I,i,j-1,k) - val);
            this->get_patch(i, j-1, k, patch1);
            s[2] = (*this.*dist_func)(patch0, patch1);
            d[2] = GET(_D,i,j-1,k) + s[2];
            N[2] = 1;
            // dj = GET(_D,i,j-1,k) + std::abs(GET(_I,i,j-1,k) - val);
        }
        
        ///j+1
        if ((j<nj-1) && (GET(_Q,i,j+1,k)>=0))
        {
            // s[3] = std::abs(GET(_I,i,j+1,k) - val);
            this->get_patch(i, j+1, k, patch1);
            s[3] = (*this.*dist_func)(patch0, patch1);
            d[3] = GET(_D,i,j+1,k) + s[3];
            N[3] = 1;
            // dj = std::min(
                // dj, GET(_D,i,j+1,k) + std::abs(GET(_I,i,j+1,k) - val));
        }
        
        /// k-1
        if ((k>0) && (GET(_Q,i,j,k-1)>=0))
        {
            // s[4] = std::abs(GET(_I,i,j,k-1) - val);
            this->get_patch(i, j, k-1, patch1);
            s[4] = (*this.*dist_func)(patch0, patch1);
            d[4] = GET(_D,i,j,k-1) + s[4];
            N[4] = 1;
            // dk = GET(_D,i,j,k-1) + std::abs(GET(_I,i,j,k-1) - val);
        }
        
        ///k+1
        if ((k<nk-1) && (GET(_Q,i,j,k+1)>=0))
        {
            // s[5] = std::abs(GET(_I,i,j,k+1) - val);
            this->get_patch(i, j, k+1, patch1);
            s[5] = (*this.*dist_func)(patch0, patch1);
            d[5] = GET(_D,i,j,k+1) + s[5];
            N[5] = 1;
            // dk = std::min(
                // dk, GET(_D,i,j,k+1) + std::abs(GET(_I,i,j,k+1) - val));
        }
        */
        
        float dist;
        
        /// method 1
        if (_method==1)
        {
            float S = 0;
            for (unsigned int u=0; u<_nsize; u++)
                S += N[u]/(s[u]+(float)1e-10);
                
            dist = 0;
            for (unsigned int u=0; u<_nsize; u++)
                dist += N[u]*d[u]/(s[u]+(float)1e-10)/S;
                
        }
        
        ///method 0
        else 
            dist = *std::min_element(d, d+_nsize);
        
            
        delete[] patch0;
        delete[] patch1;
        
        delete[] d;
        delete[] s;
        delete[] N;
        
        return dist;
    }
    
    
    float dist(unsigned int i1, unsigned int j1, unsigned int k1, 
               unsigned int i2, unsigned int j2, unsigned int k2, 
               float offset)
    {
        unsigned int diam = (2*_radius+1);
        unsigned int diam3 = diam*diam*diam;
        float *patch1 = new float[diam3];
        float *patch2 = new float[diam3];
        
        float (DistancePatchAniso::*dist_func)
            (float const *, float const *) const = NULL;
        dist_func = &DistancePatchAniso::dist_mean;
        
        float _offset = offset *
             (float)std::sqrt(1.*(i1!=i2) + 1.*(j1!=j2) + 1.*(k1!=k2));
        
        this->get_patch(i1, j1, k1, patch1);
        this->get_patch(i2, j2, k2, patch2);
        return (*this.*dist_func)(patch1, patch2) + _offset;
        
    }
    
    DistancePatchAniso( unsigned int const size[3],
              float const *I, int const *Q, float const *D, 
              int const (*neigh)[3], unsigned int neigh_size, int method=0, 
              unsigned int rad=1, unsigned int nbins=100)
              
    : _I(I), _Q(Q), _D(D), 
      _neigh(neigh), _nsize(neigh_size),
      ni(size[0]), nj(size[1]), nk(size[2]), 
      _method(method), _radius(rad) {}
    
  private:
    int const *_Q;
    float const *_D;
    float const *_I;
    int const (*_neigh)[3];
    unsigned int _nsize;
    unsigned int ni,nj,nk;
    int _method;
    unsigned int _radius;
    unsigned int _nbins;
};

#endif
