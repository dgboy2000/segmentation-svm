
#include <iostream>
#include <string>
#include <cmath>
#include <typeinfo>
#include "src/MRFEnergy.h"
//#include "src/typeGeneral.h"
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

///-----------------------------------------------------------------------------
//extern __declspec(dllexport) 
double trw_general(
    int nnodes,           // number of nodes
    int nlabels,          // number of labels
    double const* unary,  // unary potentials
    int npairs,           // number of pairs
    int const *pairs,     // pairs (ith pair = pairs[2*i], pairs[2*i+1])
    double const *bpots,  // binary potentials
    // int* solution,        // return array
    Allocator allocator,  // allocator function for return arrays
    int max_iter,         // max number of iterations
    bool compute_min_marginals=false,   // compute min_marginals
    bool randomize_order=false, // 
    bool use_bp=false,          // belief propagation 
    bool verbose=false          // print debug info
    )
{
    MRFEnergy<TypeGeneral>* mrf;
    MRFEnergy<TypeGeneral>::NodeId* nodes;
    MRFEnergy<TypeGeneral>::Options options;
    TypeGeneral::REAL energy, lowerBound;

    mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
    nodes = new MRFEnergy<TypeGeneral>::NodeId[nnodes];

    // construct energy
    TypeGeneral::REAL * U = new TypeGeneral::REAL[nlabels];
    for (int i=0; i < nnodes; i++)
    {
        for (int l=0; l < nlabels; l++)
        {
            U[l] = unary[i*nlabels + l];
        }
    
        nodes[i] = mrf->AddNode(
            TypeGeneral::LocalSize(nlabels), 
            TypeGeneral::NodeData(U)
            );
    }
    delete[] U;
    
    TypeGeneral::REAL * E = new TypeGeneral::REAL[nlabels*nlabels];
    for (int i=0; i < npairs; i++)
    {
        for (int l1 = 0; l1 < nlabels; l1++)
        for (int l2 = 0; l2 < nlabels; l2++)
        {{
            E[l1 + l2*nlabels] = bpots[i*nlabels*nlabels + l1 + l2*nlabels];
        }}
        mrf->AddEdge(
            nodes[pairs[i*2]], 
            nodes[pairs[i*2+1]], 
            TypeGeneral::EdgeData(TypeGeneral::GENERAL, E)
            );
    }

    // Function below is optional - it may help if, for example, nodes are added in a random order
    if (randomize_order) mrf->SetAutomaticOrdering();

    /////////////////////// TRW-S algorithm //////////////////////
    if (verbose) options.m_printMinIter = 0;
    else options.m_printMinIter = max_iter+1;
    
    options.m_iterMax = max_iter; // maximum number of iterations
    
    if (use_bp)
    {
        mrf->Minimize_BP(options, energy);
    }
    else if (compute_min_marginals)
    {
        unsigned int size[2] = {nnodes,nlabels};
        TypeGeneral::REAL *min_marginals = (TypeGeneral::REAL*) allocator(
            2, size, typeid(TypeGeneral::REAL).name(), "min_marginals");
            
        mrf->Minimize_TRW_S(options, lowerBound, energy, min_marginals);
    }
    else 
    {
        mrf->Minimize_TRW_S(options, lowerBound, energy);
    }

    // read solution
    unsigned int size[1] = {nnodes};
    int *solution = (int*) allocator(1, size, "int", "solution");
    for (int i = 0; i<nnodes; i++)
    {
        solution[i] = mrf->GetSolution(nodes[i]);
    }

    delete mrf;
    delete[] nodes;
    
    return energy;
}



#ifdef __cplusplus
}
#endif
