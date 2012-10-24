//#define DLLEXPORT __declspec(dllexport)
/*
#ifdef MRF_API_DLL
  #ifdef MRF_API_EXPORT
    #define MRF_API __declspec(dllexport)
  #else
    #define MRF_API __declspec(dllimport)
  #endif
#else
  #define MRF_API //extern
#endif
*/

#include <iostream>
#include <string>
#include <cmath>

#include "src/CV_Fast_PD.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef float T;


///-----------------------------------------------------------------------------
//extern DLLEXPORT T fastPD(
extern  T fastPD(
    int nnodes,           // number of nodes
    int nlabels,          // number of labels
    T const* unary,       // unary potentials
    int npairs,           // number of pairs
    int const * pairs,    // pairs (ith pair = pairs[2*i], pairs[2*i+1])
    T const * binary,     // binary potentials
    T const * wedges,     // edge weighting
    int* results,         // return array
    int max_iter,         // max number of iterations
    bool debug            // print debug info
    )
{

    T energy = 0;
  
    typedef CV_Fast_PD::Real Real;

    if (debug) std::cout << "prepare data for optimization\n";
    /// parameters
    int numpoints = (int)nnodes;
    int numlabels = (int)nlabels;
    int numpairs = (int)npairs;
    
    Real *lcosts 
        = new Real[numpoints*numlabels];

    int *pairs_ = new int[2*numpairs];
    
    Real *dist
        = new Real[numlabels*numlabels];
    
    Real *wcosts
        = new Real[numpairs];

    /// fill matrices: single potentials
    for (int i = 0; i < nnodes; ++i)
    {
        for (int l = 0; l < numlabels; ++l)
        {
            lcosts[l*numpoints+i] = (Real)unary[i*numlabels+l];
        }
    }
    /// binary potentials
    for (int i = 0; i < numlabels; i++)
    {
        for (int j = 0; j < numlabels; j++)
        {
            dist[i*numlabels + j] = (Real)binary[i*numlabels + j];
        }
    }
    /// pairs
    for (int i = 0; i < numpairs; ++i)
    {
        pairs_[i*2] = (int)pairs[i*2];
        pairs_[i*2+1] = (int)pairs[i*2+1];
        wcosts[i] = (Real)wedges[i];
    }
    
    /// run optimizer
    if (debug) std::cout << "optimize MRF\n";
    CV_Fast_PD optimizer( 
        numpoints,
        numlabels,
        lcosts,
        numpairs,
        pairs_,
        dist,
        max_iter,
        wcosts
         );
    
    if (debug) std::cout << "run fastPD optimizer...\n";
    optimizer.run();

    /// get results
    
    for (int i = 0; i < nnodes; ++i)
    {
        results[i] = (int)optimizer._pinfo[i].label;
        energy += lcosts[results[i]*numpoints + i];
    }
    
    delete[] lcosts;
    delete[] pairs_;
    delete[] dist;
    delete[] wcosts;

    if (debug) std::cout << "done optimize MRF: energy = " << energy << "\n";    
    return energy;
    
}   /// end optim_MRF_grid()


#ifdef __cplusplus
}
#endif
