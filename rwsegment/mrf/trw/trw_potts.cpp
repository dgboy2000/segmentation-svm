
#include <iostream>
#include <string>
#include <cmath>

#include "src/MRFEnergy.h"

#ifdef __cplusplus
extern "C" {
#endif

///-----------------------------------------------------------------------------
//extern __declspec(dllexport) 
double trw_potts(
    int nnodes,           // number of nodes
    int nlabels,          // number of labels
    double const* unary,  // unary potentials
    int npairs,           // number of pairs
    int const *pairs,     // pairs (ith pair = pairs[2*i], pairs[2*i+1])
    double const *wpairs, // binary potentials
    int* solution,        // return array
    int max_iter,         // max number of iterations
    bool randomize_order, //
    bool use_bp,          // belief propagation 
    bool verbose          // print debug info
    )
{
    MRFEnergy<TypePotts>* mrf;
    MRFEnergy<TypePotts>::NodeId* nodes;
    MRFEnergy<TypePotts>::Options options;
    TypePotts::REAL energy, lowerBound;

    mrf = new MRFEnergy<TypePotts>(TypePotts::GlobalSize(nlabels));
    nodes = new MRFEnergy<TypePotts>::NodeId[nnodes];

    // construct energy
    TypePotts::REAL * U = new TypePotts::REAL[nlabels];
    for (int i=0; i < nnodes; i++)
    {
        for (int l=0; l < nlabels; l++)
        {
            U[l] = unary[i*nlabels + l];
        }
    
        nodes[i] = mrf->AddNode(
            TypePotts::LocalSize(), 
            TypePotts::NodeData(U)
            );
    }
    delete[] U;
    
    
    
    /// pairs
    for (int i = 0; i < npairs; ++i)
    {
        mrf->AddEdge(
            nodes[pairs[i*2]], nodes[pairs[i*2+1]], 
            TypePotts::EdgeData(wpairs[i])
            );
    }
    
    // TypeGeneral::REAL * E = new TypeGeneral::REAL[nlabels*nlabels];
    // for (int i=0; i < npairs; i++)
    // {
        // for (int l1 = 0; l1 < nlabels; l1++)
        // for (int l2 = 0; l2 < nlabels; l2++)
        // {{
            // E[l1 + l2*nlabels] = wpairs[i*nlabels*nlabels + l1 + l2*nlabels];
        // }}
        // mrf->AddEdge(
            // nodes[pairs[i*2]], 
            // nodes[pairs[i*2+1]], 
            // TypeGeneral::EdgeData(TypeGeneral::GENERAL, E)
            // );
    // }

    // Function below is optional - it may help if, for example, nodes are added in a random order
    if (randomize_order) mrf->SetAutomaticOrdering();

    /////////////////////// TRW-S algorithm //////////////////////
    if (verbose) options.m_printMinIter = 0;
    else options.m_printMinIter = max_iter+1;
    
    options.m_iterMax = max_iter; // maximum number of iterations
    
    if (use_bp) mrf->Minimize_BP(options, energy);
    else mrf->Minimize_TRW_S(options, lowerBound, energy);

    // read solution
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
