#include <iostream>
#include <queue>
#include <cmath>
#include <algorithm>

#include "common.h"
#include "time_standard.h"
#include "time_aniso.h"
#include "time_aniso_sobel.h"
// #include "time_aniso2.h"





///-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

//extern __declspec(dllexport) 


/** Fast marching function
 */
int fast_marching_3d_general(
    unsigned int size[3], float const *image, 
    unsigned int npoints, unsigned int const *points,
    Allocator allocator,
    unsigned int const *plabels=NULL,
    bool *mask=NULL,
    unsigned int heap_size=1000,
    float offset=0.,
    bool connect26=false,
    int method=0,
    bool debug=false
    )
    
{
    /// options
    bool return_parents = true;
    bool return_inter = true;
    bool return_edges = true;
    bool return_edge_size = true;
    
    /// allocate D (arrival time), Q (label), parents and intersections
    float *D = (float*) allocator(3, size, "float32", "distances");
    int *Q = (int*) allocator(3, size, "int32", "labels");
    int *parents = NULL;
    int *inters = NULL;
    if (return_parents) parents = (int*) allocator(3, size, "int32", "parents");
    if (return_inter) 
        inters = (int*) allocator(3, size, "int32", "intersections");
    
    /// edges and intersections
    EdgeData<float> edges(npoints);
    EdgeData<int> nedges(npoints);
    
    
    /// image size
    unsigned int ni = size[0];
    unsigned int nj = size[1];
    unsigned int nk = size[2];
    
    
    
    /// neighborhood
    int neighborhood_type = connect26 ? 1 : 0;
    unsigned int nsize = neighborhood_size[neighborhood_type];
    int (*neigh)[3] = (neighborhood_type==1) ? neighborhood26 : neighborhood6;

    
    /// init labels and distance
    for (int iq = ni*nj*nk - 1; iq >= 0; iq--) 
    {
        Q[iq] = -2;
        D[iq] = 1e10;
        if (return_parents) parents[iq] = 0;
    }
    
    /// Distance function class
    ArrivalTime * time_function = NULL;
    switch(method){
        case 0:
            time_function = new ArrivalTime_standard(
                size, offset, image, Q, D, neighborhood_type);
            break;
        case 1:
            time_function = new ArrivalTime_aniso(
                size, offset, image, Q, D, neighborhood_type);
            break;
        case 2:
            time_function = new ArrivalTime_sobel(
                size, offset, image, Q, D, neighborhood_type);
            break;
    }
    /// delta
    float delta = time_function->compute_delta(heap_size);
    
    /// heap
    unsigned int top = 0;
    std::queue<Cell> *H = new std::queue<Cell>[heap_size+1];
    for (unsigned int ip=0; ip<npoints; ++ip)
    {
        unsigned int i = points[ip*3];
        unsigned int j = points[ip*3+1];
        unsigned int k = points[ip*3+2];
        
        SET(D,i,j,k,0.0);
        SET(Q,i,j,k,-1);
        
        unsigned int label;
        if (plabels==NULL) label = ip;
        else label = plabels[ip];
        
        Cell c( GET(Q,i,j,k), GET(D,i,j,k), 
                label, i, j, k, GET(image,i,j,k) );
            
        H[top].push(c); /// push cell (copy) at the end of the top queue
    }
    
    
    /// loop: while heap not empty
    unsigned int n_empty = 0;
    int ipath = 1; // path index
    while (1)
    {
        if (H[top].empty())
        {
            n_empty +=1;
            top = (top+1)%(heap_size+1);
            
            /// stop condition: all queues are empty
            if (n_empty == heap_size+1) break;
            else continue;
        }
        // std::cout <<"TOP: " << top << "\n";
        n_empty = 0;
        
        /// extract front cell from top queue
        Cell const &c = H[top].front();
        int state = c.q;
        unsigned int label = c.label;
        unsigned int ci = c.i, cj = c.j, ck = c.k;
        float cdist = c.d;
        
        if (state >= 0)
        {
            /// skip if already frozen
            H[top].pop(); /// remove cell from queue
            continue;
        }
        else
        {
            /// freeze c
            c.q = label; 
            H[top].pop(); /// remove cell from queue
            
        }
               
        /// for each neighbor of v
        for (unsigned int in=0; in<nsize; ++in)
        {
            int i = ci + neigh[in][0];
            int j = cj + neigh[in][1];
            int k = ck + neigh[in][2];

            if ((i<0) || (i>=(int)ni) || 
                (j<0) || (j>=(int)nj) || 
                (k<0) || (k>=(int)nk) )
                continue;

            /// skip if outside mask
            if ((mask!=NULL)&&(!GET(mask,i,j,k)))
                continue;
                
            /// get state of neighbor
            int _state = GET(Q, i, j, k);
            
            /// if not frozen
            if (_state < 0)
            {
                float time = (*time_function)(
                    (unsigned int)i,
                    (unsigned int)j,
                    (unsigned int)k, ci, cj, ck);
                    
                    
                /// save distance if shorter
                float & d = GET(D,i,j,k);
                if ((_state==-1) && (time > (d - 1e-10))) continue;
                
                /// mark as narrow band
                else if (_state==-2) SET(Q, i, j, k, -1);
                
                SET(D, i, j, k, time);
                
                /// parents
                if (return_parents) 
                {
                    int iparent = GETID(ci, cj, ck);
                    SET(parents, i, j, k, iparent);
                }
                
                /// add to the heap
                int h_index = int(time/delta)%(heap_size+1);
                Cell _c(GET(Q,i,j,k), GET(D,i,j,k), label, i, j, k);
                H[h_index].push(_c);
                
            } /// end if (state not frozen)
            
            /// fill edges and intersection
            else if (_state!=label)
            {
                if (return_edges)
                {
                    /// shortest path
                    float time = (*time_function).compute_interval(
                        (unsigned int)i,
                        (unsigned int)j,
                        (unsigned int)k,
                        ci, cj, ck);
                        
                    float d_ijk = GET(D,i,j,k);
                    float value = cdist + d_ijk + time;
                    

                    if (!edges.has(_state, label))
                    {
                        edges.insert(_state, label, value);
                       // edges.insert(_state, label, 0);
                        if (return_edge_size) nedges.insert(_state, label, 0);
                    }
                    // edges.insert(_state, label, \
                                 // edges.get(_state, label) + value);
                    if (return_edge_size) 
                        nedges.insert(_state, label, 
                                      nedges.get(_state, label) + 1);
                }
                if (return_inter)
                {
                    /// intersections
                    int i1 = GET(inters, i, j, k);
                    int i2 = GET(inters, ci, cj, ck);
                        
                    if ((i1==0)&&(i2==0))
                    {
                        SET(inters, i, j, k, ipath);
                        SET(inters, ci, cj, ck, ipath);
                        ipath += 1;
                    }
                    else if (i1==0)
                    {
                        SET(inters, i, j, k, i2);
                    }
                    else
                    {
                        SET(inters, ci, cj, ck, i1);
                    }
                }
            }
            
        } /// end for (each neighbor)
        
    } /// end while (heap not empty)

    /// fill edge data
    if (return_edges)
    {
        int nedge = edges.number_of_edges();
        unsigned int eshape[] = {nedge, 2};
        unsigned int dshape[] = {nedge};
        int *E = (int*) allocator(2, eshape, "int32", "edges");
        float *vE = (float*) allocator(1, dshape, "float32", "edge_values");
        edges.get_data(vE);
        edges.get_edges(E);
        
        if (return_edge_size) 
        {
            int *nE = (int*) allocator(1, dshape, "int32", "edge_size");
            nedges.get_data(nE);
        }
    }
    
    delete time_function;
    delete[] H;
    
    return 0;
    
} /// end fast_marching_3d_aniso()
///-----------------------------------------------------------------------------
#ifdef __cplusplus
}
#endif
