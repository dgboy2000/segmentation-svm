#ifndef __distance_aniso_sobel_h__
#define __distance_aniso_sobel_h__

#include <cmath>
#include <algorithm>

#include "common.h"


///-----------------------------------------------------------------------------
class ArrivalTime_sobel: public ArrivalTime
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
    
    /** duration between two pixels
     *  sobel filter
     *      1  2  1
     *      2  4  2
     *      1  2  1
     *
     *     -1 -2 -1
     *     -2 -4 -2
     *     -1 -2 -1
     *
     */
    
    virtual float compute_interval(unsigned int i1,
                                   unsigned int j1,
                                   unsigned int k1,
                                   unsigned int i2,
                                   unsigned int j2,
                                   unsigned int k2) const
    {
        unsigned int nsize = 26;
        int (*neigh)[3] = neighborhood26;
        
        int dir[3]; 
        dir[0] = (int)i2 - (int)i1;
        dir[1] = (int)j2 - (int)j1;
        dir[2] = (int)k2 - (int)k1;  
        
        // std::cout << "----------\n";
        // std::cout << "dir=" << dir[0] << " " << dir[1] << " " << dir[2] << "\n";
        // std::cout << "ijk1=" << i1 << " " << j1 << " " << k1 << "\n";
        
        /// does not work in connectivity 26
        int diag = std::abs(dir[0]) + std::abs(dir[1]) + std::abs(dir[2]) - 1;
        
        float time = 0;
        for (unsigned int in=0; in<nsize; in++)
        {
            int _i = neigh[in][0] + i2;
            int _j = neigh[in][1] + j2;
            int _k = neigh[in][2] + k2;
            if ((_i>=0) && (_i<(int)ni) &&
                (_j>=0) && (_j<(int)nj) &&
                (_k>=0) && (_k<(int)nk))
            {
                // std::cout << "nei=" << neigh[in][0] << " "
                                    // << neigh[in][1] << " "
                                    // << neigh[in][2] << "\n";
                // std::cout << "prod=" << neigh[in][0]*dir[0] + neigh[in][1]*dir[1] + neigh[in][2]*dir[2] << "\n";
            
                /// target
                if ((neigh[in][0]*dir[0]  +
                     neigh[in][1]*dir[1]  + 
                     neigh[in][2]*dir[2]) == 0)
                {
                    int val = 2 - std::abs(neigh[in][0]) -
                                  std::abs(neigh[in][1]) - 
                                  std::abs(neigh[in][2]);
                    time += (float)std::pow(2., val) * GET(_I,_i,_j,_k);

                    // std::cout << "val=" << (float)std::pow(2., val) << "\n";
                }
                
                /// origin
                else if ((neigh[in][0]*dir[0] +
                          neigh[in][1]*dir[1] +
                          neigh[in][2]*dir[2]) == -1)
                {
                    int val = 2 - std::abs(neigh[in][0]) -
                                  std::abs(neigh[in][1]) - 
                                  std::abs(neigh[in][2]) + 1*(diag==0);
                    time -= (float)std::pow(2., val) * GET(_I,_i,_j,_k);
                    
                    // std::cout << "val=" << (float)std::pow(2., val) << "\n";
                }
                
                /// test
                else if ((neigh[in][0]*dir[0] +
                          neigh[in][1]*dir[1] +
                          neigh[in][2]*dir[2]) < -1)
                {
                    int val = 2 - std::abs(neigh[in][0]) -
                                  std::abs(neigh[in][1]) - 
                                  std::abs(neigh[in][2]) + diag;
                    time -= (float)std::pow(2., val) * GET(_I,_i,_j,_k);
                    
                    // std::cout << "val=" << (float)std::pow(2., val) << "\n";
                }
                
            }
        }
        time += 4 * GET(_I, i2, j2, k2);
        
        return (float)std::abs(time)/(16 - 4*(diag>0)) + _offset;
        
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
    ArrivalTime_sobel(unsigned int const size[3], float offset,
              float const *I, int const *Q, float const *D, 
              int neighborhood_type=0)
    : ArrivalTime(size, offset, I, Q, D, neighborhood_type), _max_gap(0)
    { std::cout << "Using Arrival Time Anisotropic method with sobel filter\n";}
  
  private:
    float _max_gap;
  
};

#endif
