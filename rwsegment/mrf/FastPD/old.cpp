
///-----------------------------------------------------------------------------
class GraphMatchingCostFunction:
    public CV_Fast_PD_cf::CostFunction
{

  public:

    typedef CV_Fast_PD_cf::Real Real;
  
    /** constructor
     *
     */
    GraphMatchingCostFunction(
        int nnodes1, T const *dm1,
        int ncandidates, int const *candidates,
        int nnodes2,T const *dm2,
        int npairs, int *pairs, T const *binary)
    {
        _nnodes1 = nnodes1;
        _nnodes2 = nnodes2;
        _npairs = npairs;
        _ncandidates = ncandidates;
        _dm1 = dm1;
        _dm2 = dm2;
        _pairs = pairs;
        _candidates = candidates;
        _binary = binary;
        
    } /// constructor

    
    /** Access function 
     *
     */
    virtual Real computeDistance(int pair_index,
                                 int label_1, int label_2)
    {
        int node1 = _pairs[pair_index*2];
        int node2 = _pairs[pair_index*2+1];
        
        int cand1 = _candidates[node1*_ncandidates + label_1];
        int cand2 = _candidates[node2*_ncandidates + label_2];
        
        Real dist1 = _dm1[node1*_nnodes1 + node2];
        Real dist2 = _dm2[cand1*_nnodes2 + cand2];
        
        // Real cost =  (dist1 - dist2)*(dist1 - dist2) / ((dist1 + dist2)*(dist1 + dist2));
        
        Real cost = std::fabs(dist1-dist2);
        //Real cost = std::fabs(dist1-dist2) + (dist1-dist2)*(dist1-dist2);
        
        // std::cout 
            // << "binary: \n"
            // << "\t pair #" << pair_index << "\n"
            // << "\t l1:" << label_1 << ", l2: " << label_2 << "\n"
            // << "\t node1: " << node1 << ", cand1: " << cand1 << "\n"
            // << "\t node2: " << node2 << ", cand2: " << cand2 << "\n"
            // << "\t dist1: " << dist1 << ", dist2: " << dist2 << "\n"
            // << "\t cost: " <<  cost << "\n"
            // ;
            
        return cost + _binary[_nnodes2*cand1 + cand2];
        
    
    } /// computeDistance()
    
    int _nnodes1, _nnodes2, _npairs, _ncandidates;
    int *_pairs;
    int const *_candidates;
    T const *_dm1;
    T const *_dm2;
    T const * _binary;
};



///-----------------------------------------------------------------------------
extern MRF_API T graph_matching(
    int nnodes1,           // number of nodes in graph 1
    T const* dm1,            // distance matrix of graph 1
    int ncandidates,
    int const * candidates,
    T const *unary,         // unary matching cost
    int nnodes2,           // number of nodes in graph 2
    T const* dm2,            // distance matrix of graph 2
    T const *binary,
    int npairs,           // number of pairs
    int const * pairs,    // pairs (ith pair = pairs[2*i], pairs[2*i+1])
    int* results,              // return array
    char const * method,     // method ('fastPD' or 'TRW')
    int max_iter,           // max number of iterations
    bool debug,              // print debug info
    int *init
    )
{

    T energy = 0;
  
    if (std::string(method) == "fastPD")
    {
        typedef CV_Fast_PD::Real Real;
    
        if (debug) std::cout << "prepare data for optimization\n";
        /// parameters
        int numpoints = (int)nnodes1;
        int numlabels = (int)nnodes2;
        int numpairs = (int)npairs;
        
        Real *lcosts 
            = new Real[numpoints*numlabels];

        int *pairs_ = new int[2*numpairs];
        
        Real *dist
            = new Real[numlabels*numlabels];
        
        Real *wcosts
            = new Real[numpairs];
    
        /// fill matrices: single potentials
        // std::cout << "singles \n";
        for (int i = 0; i < numpoints; ++i)
        {
            // std::cout << "node " << i << ": ";
            for (int l = 0; l < numlabels; ++l)
            {
                lcosts[l*numpoints+i] = unary[i*numlabels+l];
                // std::cout << "label[" << l << "]=" << lcosts[l*numpoints+i] << " ";
            }
        }
        // std::cout << "\n";
        
        // binary potentials: cost function
        GraphMatchingCostFunction cost_function(
            nnodes1, dm1, 
            ncandidates, candidates, 
            nnodes2, dm2, 
            npairs, pairs_,
            binary );
        
        
        /// pairs
        // std::cout << "pairs:\n";
        for (int i = 0; i < numpairs; ++i)
        {
            pairs_[i*2] = (int)pairs[i*2];
            pairs_[i*2+1] = (int)pairs[i*2+1];
            // std::cout << "pair " << i << ": (" << pairs[i*2] << "," 
                // << pairs[i*2+1] << ")=" << wedges[i] << "\n";
        }
        // std::cout << "\n";
        
        
        /// run optimizer
        if (debug) std::cout << "optimize MRF\n";
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
        
        double enn = 0;
        for (int i = 0; i < numpoints; ++i)
        {
            results[i] = (int)optimizer._pinfo[i].label;
            enn += unary[results[i] + numlabels*i];
            // energy += lcosts[results[i]*numpoints + i];
            //std::cout << energy << "\n";
        }
        // std::cout << "energy nodes: " << energy << "\n";
        
        
        double enp = 0;
        for (int i = 0; i < npairs; i++)
        {
            double enp_ = cost_function.computeDistance( i,results[pairs[i*2]],results[pairs[i*2+1]]);
            // std::cout << "pair " << i << ": " << pairs[i*2] << " " << pairs[i*2+1] << ", cost = " << enp_ << "\n";
            enp += enp_;
        }
        // std::cout << "energy pairs: " << enp << "\n";
        energy = enp + enn;
    
        delete[] lcosts;
        delete[] pairs_;
    } /// end optim fastPD
    
    
    
    if (debug) std::cout << "done optimize MRF: energy = " << energy << "\n";
    
    return energy;
}   /// end match_graph()