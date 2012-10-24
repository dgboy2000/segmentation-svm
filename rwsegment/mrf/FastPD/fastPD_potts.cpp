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
class CostFunction_Potts:
    public CV_Fast_PD_cf::CostFunction
{

  public:
    
    typedef CV_Fast_PD_cf::Real Real;
  
    /// constructor
    CostFunction_Potts(
        int nlabel, int const *labels, int const * pairs, T const *wpairs)
    { 
        _nlabel = nlabel; 
        _pairs = pairs;
        _labels = labels; 
        _wpairs = wpairs;
    }
    
    virtual Real computeDistance(int pair_index,
                                 int label_1, int label_2)
    { 
        if (_labels==NULL)
            return _wpairs[pair_index]*(label_1!=label_2);
        else
        {
            int n1 = _pairs[pair_index*2];
            int n2 = _pairs[pair_index*2 + 1];
            int l1 =  _labels[n1*_nlabel + label_1];
            int l2 =  _labels[n2*_nlabel + label_2];
            // std::cout << "pair index=" << pair_index << "\n";
            // std::cout << "l1=" << l1 << "\n";
            // std::cout << "l2=" << l2 << "\n";
            // std::cout << "cost ?" << ((l1 <= 0)||(l2 <= 0)||(l1 != l2))*_wpairs[pair_index] << "\n";
            return ((l1 <= 0)||(l2 <= 0)||(l1 != l2))?(_wpairs[pair_index]):0.;
                
        }
    }
    
 private:
    int _nlabel;
    int const * _labels;
    int const * _pairs;
    T const * _wpairs;
};


///-----------------------------------------------------------------------------
//extern DLLEXPORT T fastPD_potts(
extern  T fastPD_potts(
    int nnodes,           // number of nodes
    int nlabels,          // number of labels
    T const *unary,       // unary potentials
    int const *labels,    // labels (if more labels than nlabels)
    int npairs,           // number of pairs
    int const *pairs,     // pairs (ith pair = pairs[2*i], pairs[2*i+1])
    T const * wedges,     // edge weighting
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

    CostFunction_Potts cost_function(nlabels, labels, pairs, wedges);
    
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
