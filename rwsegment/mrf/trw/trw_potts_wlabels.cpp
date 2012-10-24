
#include <iostream>
#include <string>
#include <cmath>

#include "src/MRFEnergy.h"

#ifdef __cplusplus
extern "C" {
#endif

///-----------------------------------------------------------------------------
//extern __declspec(dllexport) 
double trw_potts_wlabels(
    int nnodes,           // number of nodes
    int nlabels,          // max number of labels
    double const* unary,  // unary potentials
    int const* labels,    // unary potentials
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
    MRFEnergy<TypeGeneral>* mrf;
    MRFEnergy<TypeGeneral>::NodeId* nodes;
    MRFEnergy<TypeGeneral>::Options options;
    TypeGeneral::REAL energy, lowerBound;

    mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
    nodes = new MRFEnergy<TypeGeneral>::NodeId[nnodes];

    int *local_size = new int[nnodes];
    
    // std::cout << "a\n" 
        // << "nnodes=" << nnodes << "\n"
        // << "nlabels=" << nlabels << "\n"
        // << "npairs=" << npairs << "\n"
        // ;
        
    // construct energy
    TypeGeneral::REAL * U = new TypeGeneral::REAL[nlabels];
    for (int i=0; i < nnodes; i++)
    {
        int ilabel = 0;
        for (int l=0; l < nlabels; l++)
        {
            if (labels[i*nlabels + l] >= 0)
            {
                U[ilabel] = unary[i*nlabels + l];
                ilabel += 1;
            }
        }
        // std::cout << "\tnode=" << i << ", nlabel="<< ilabel << "\n"; 
    
        local_size[i] = ilabel;
        nodes[i] = mrf->AddNode(
            TypeGeneral::LocalSize(ilabel), 
            TypeGeneral::NodeData(U)
            );
    }
    delete[] U;
    
    // std::cout << "b\n";
    
    for (int i=0; i < npairs; i++)
    {
        int n1 = pairs[i*2];
        int n2 = pairs[i*2 + 1];
        int s1 = local_size[n1];
        int s2 = local_size[n2];
        // std::cout << "\t\edge=" << i << ", n1=" << n1 << ", n2=" << n2 << "\n";
        // std::cout << "s1=" << s1 << ", s2=" << s2 << "\n";
        
        TypeGeneral::REAL * E = new TypeGeneral::REAL[s1*s2];
        int ilabel1=0;
        for (int l1 = 0; l1 < nlabels; l1++)
        {
            int lab1 = labels[n1*nlabels + l1];
            if (lab1<0) continue;

            // std::cout << "\tlab1=" << lab1 << ", ilab=" << ilabel1 << "\n";
            int ilabel2 = 0;
            for (int l2 = 0; l2 < nlabels; l2++)
            {
                int lab2 = labels[n2*nlabels + l2];
                if (lab2<0) continue;
                
                
                // std::cout << "\tlab2=" << lab2 << ", ilab=" << ilabel2 << "\n";
                // std::cout << "\twpair=" << wpairs[i] << "\n";
                if (lab1==lab2) E[ilabel1 + ilabel2*s1] = 0;
                else E[ilabel1 + ilabel2*s1] = wpairs[i];
                ilabel2 +=1;
            }
            ilabel1 += 1;
        }
        mrf->AddEdge(
            nodes[pairs[i*2]], 
            nodes[pairs[i*2+1]], 
            TypeGeneral::EdgeData(TypeGeneral::GENERAL, E)
            );
            
        delete[] E;
    }
    // std::cout << "c\n";

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
    delete[] local_size;
    
    return energy;
}



#ifdef __cplusplus
}
#endif
