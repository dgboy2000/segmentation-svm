
#include <iostream>
#include <string>
#include <cmath>

#include "src/MRFEnergy.h"
#include "src/typeBinary.h"

#ifdef __cplusplus
extern "C" {
#endif

///-----------------------------------------------------------------------------
//extern __declspec(dllexport) 
double trw_binary(
    int nnodes,           // number of nodes
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
    MRFEnergy<TypeBinary>* mrf;
	MRFEnergy<TypeBinary>::NodeId* nodes;
	MRFEnergy<TypeBinary>::Options options;
	TypeBinary::REAL energy, lowerBound;

	mrf = new MRFEnergy<TypeBinary>(TypeBinary::GlobalSize());
	nodes = new MRFEnergy<TypeBinary>::NodeId[nnodes];

	// construct energy
    for (int i=0; i < nnodes; i++)
    {
        nodes[i] = mrf->AddNode(
            TypeBinary::LocalSize(), 
            TypeBinary::NodeData(
                unary[i*2], unary[i*2+1]
                )
            );
    }
    
    for (int i=0; i < npairs; i++)
    {
        mrf->AddEdge(
            nodes[pairs[i*2]], 
            nodes[pairs[i*2+1]], 
            TypeBinary::EdgeData(
                wpairs[i*4], 
                wpairs[i*4+1], 
                wpairs[i*4+2], 
                wpairs[i*4+3]
                )
            );
    }

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

    // delete mrf;
    // delete[] nodes;
    
    return energy;
}



#ifdef __cplusplus
}
#endif
