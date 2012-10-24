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

#include "src/CV_Fast_PD_costfunction.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef float T;

///----------------------------------------------------------------------------
class GraphMatchingCallbackCostFunction:
    public CV_Fast_PD_cf::CostFunction
{

  public:
    
    typedef CV_Fast_PD_cf::Real Real;
  
    GraphMatchingCallbackCostFunction(T (*fn)(int,int,int))
    { _fn = fn; }
    
    virtual Real computeDistance(int pair_index,
                                 int label_1, int label_2)
    { 
        return (*_fn)(pair_index,label_1,label_2);
    }
    
 private:
    T (*_fn)(int,int,int);
    
};
///----------------------------------------------------------------------------

class GraphMatchingCallbackCostFunction_:
    public CV_Fast_PD_cf::CostFunction
{

  public:
    typedef CV_Fast_PD_cf::Real Real;
    virtual Real computeDistance(int pair_index,
                                 int label_1, int label_2) {return 0;}
};

///-----------------------------------------------------------------------------
//extern DLLEXPORT T fastPD_callback(
extern T fastPD_callback(
    int nnodes,           // number of nodes
    int nlabels,          // number of labels
    T const *unary,       // unary potentials
    int npairs,           // number of pairs
    int const *pairs,     // pairs (ith pair = pairs[2*i], pairs[2*i+1])
    T (*costfunction)(int,int,int),
    int *results,         // return array
    int *init,            // initial state
    int max_iter,         // max number of iterations
    bool debug            // print debug info
    )
{

    typedef CV_Fast_PD_cf::Real Real;

    if (debug) std::cout << "prepare data for optimization\n";
    
    /// parameters
    int numpoints = (int)nnodes;
    int numlabels = (int)nlabels;
    int numpairs = (int)npairs;
    T energy = 0;
    
    Real *lcosts 
        = new Real[numpoints*numlabels];
    int *pairs_ = new int[2*numpairs];


    /// fill matrices: single potentials
    for (int i = 0; i < numpoints; ++i)
    {
        for (int l = 0; l < numlabels; ++l)
        {
            lcosts[l*numpoints+i] = unary[i*numlabels+l];
        }
    }

    GraphMatchingCallbackCostFunction cost_function(costfunction);
    // GraphMatchingCallbackCostFunction_ cost_function;
    
    /// pairs
    for (int i = 0; i < numpairs; ++i)
    {
        pairs_[i*2] = (int)pairs[i*2];
        pairs_[i*2+1] = (int)pairs[i*2+1];
    }
    
    
    /// run optimizer
    if (debug) std::cout 
        << "optimize MRF\n";
        
    CV_Fast_PD_cf optimizer( 
        numpoints,
        numlabels,
        lcosts,
        numpairs,
        pairs_,
        &cost_function,
        max_iter,
        init
        );
    
    if (debug) std::cout << "run fastPD optimizer...\n";
    optimizer.run();

    /// get results
    
    T enn = 0;
    for (int i = 0; i < numpoints; ++i)
    {
        results[i] = (int)optimizer._pinfo[i].label;
        enn += unary[results[i] + numlabels*i];
    }
    
    T enp = 0;
    for (int i = 0; i < npairs; i++)
    {
        T enp_ = cost_function.computeDistance( 
            i,
            results[pairs[i*2]],
            results[pairs[i*2+1]]
            );
            
        enp += enp_;
    }
    energy = enp + enn;

    delete[] lcosts;
    delete[] pairs_;

    if (debug) std::cout << "done optimize MRF: energy = " << energy << "\n";
    
    return energy;
}   /// end match_graph()


#ifdef __cplusplus
}
#endif
