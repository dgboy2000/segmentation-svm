//#############################################################################
//#
//# CV_Fast_PD.h:
//#  Header file containing "CV_Fast_PD" class interface
//#  
//#############################################################################

#ifndef __FAST_PD_H__
#define __FAST_PD_H__

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include "graph.h"
#include "common.h"
//#include "CostFunction.h"
#include <string>
using std::string;

#define DIST(l0,l1)   (_dist[l1*_numlabels+l0])
#define UPDATE_BALANCE_VAR0(y,d,h0,h1) { (y)+=(d); (h0)+=(d); (h1)-=d; }
#define NEW_LABEL(n) ((n)->parent && !((n)->is_sink))
#define REV(a) ((a)+1)

//#############################################################################
//#
//# Classes & types
//#
//#############################################################################

//=============================================================================
// @class   CV_Fast_PD
// @author  Nikos Komodakis
//=============================================================================
//


class CV_Fast_PD
{
	public:

		typedef Graph::Real Real;

		//
		// NOTE: "lcosts" is modified by member functions of this class
		//

		// Modified by Ahmed
        CV_Fast_PD( int numpoints, int numlabels, Real *lcosts,
		           int numpairs , int *pairs   , 
		           Real *dist   , int max_iters, 
		           Real *wcosts , int *primal0 = NULL )

          
		//CV_Fast_PD(int numpoints, int numlabels, Real *lcosts, int numpairs, int *pairs, 
			//Real * wcosts, int max_iters, int  *primal0=NULL)
			// numpoints = total # of points (start from 0)
			// numlabels = total # of labels: label in [0 ..  numlabels-1]
			// lcosts = array fo single potentials: cost(point p, label l)=lcosts[l * numpoints + p] !! "lcosts" is modified by member functions of this class !!
			// numpairs = total # of pairs 
			// pairs = array of pairs: paire i: p1=pairs[2*i] p2=pairs[2*i+1]
			// wcosts = binary potentials (1-dim array)
			// max_iters = max # of iterations
			// primal0 = intitilazing labels

        {
			int i;

			//printf( "Allocating memory..." );
            //std::cout<< "    FastPD: Allocating memory..."  << std::endl;

			// Init global vars and allocate memory
			//
        	_numpoints = numpoints;
            _numlabels = numlabels;
            _numpairs  = numpairs;
			_max_iters = max_iters;
			
			_dist      = dist;
			_wcosts    = wcosts;
			_pairs     = pairs;
			//costfct	   = costFct;
			_primal0   = primal0;
			_time      =-1; 
			_APF_change_time = -1;



			if ( _numlabels >= pow(256.0f,(int) sizeof(Graph::Label)) ) 
			{
				printf( "\nChange Graph::Label type (it is too small to hold all labels)\n" );
				assert(0);
			}
				
			_children          = new Graph::node *[_numpoints];
			_source_nodes_tmp1 = new int[_numpoints]; 
			_source_nodes_tmp2 = new int[_numpoints]; 
            
			for( i = 0; i < _numpoints; i++ )
			{
				_source_nodes_tmp1[i] = -2;
				_source_nodes_tmp2[i] = -2;
			}
			
			// printf("node size=%d\n", sizeof(Graph::node));
			// printf("arc size=%d\n", sizeof(Graph::arc));
			// printf("total arc len=%d\n",_numpairs*_numlabels<<1);

			_all_nodes = new Graph::node[_numpoints*_numlabels];
			//_all_nodes = (Graph::node*) malloc(_numpoints*_numlabels*sizeof(Graph::node));
			_all_arcs  = new Graph::arc[_numpairs*_numlabels<<1];
			//_all_arcs = (Graph::arc*) malloc((_numpairs*_numlabels)<<1*sizeof(Graph::arc));
			_all_graphs = new Graph *[_numlabels];

            
			for( i = 0; i < _numlabels; i++ )
			{
				_all_graphs[i] = new Graph( &_all_nodes[_numpoints*i], &_all_arcs[_numpairs*i<<1], _numpoints, err_fun );
				fillGraph( _all_graphs[i] );
			}
			_active_list = -1;
			_pinfo = new Node_info[_numpoints];
			createNeighbors();

			_h = lcosts;
			_y = new Real[_numpairs*_numlabels];

		//	printf( "Done\n" );
        }

		~CV_Fast_PD( void )
		{
			delete[] _all_nodes; //modified by py 2011.03.11
			delete[] _all_arcs ; //modified by py 2011.03.11

			Graph::Label i;
			for( i = 0; i < _numlabels; i++ )
				delete _all_graphs[i];
			delete [] _all_graphs;

			delete [] _einfo;
			delete [] _y;

			delete [] _pairs_arr;
			delete [] _pair_info;

			delete [] _pinfo;

			delete [] _source_nodes_tmp1;
			delete [] _source_nodes_tmp2;
			delete [] _children;
		}

		void init_duals_primals( void )
		{
			//printf( "Initializing..." );

			int i;
			Graph::Label l;

			// Set initial values for primal and dual variables
			//
			for( i = 0; i < _numpoints; i++ )
			{
				_pinfo[i].label = ( _primal0 ? _primal0[i] : 0 );
				_pinfo[i].time  = -1;
				_pinfo[i].next  = -1;
				_pinfo[i].prev  = -2;
			}

			for( l = 0; l < _numlabels; l++ )
			{
				Real *cy = &_y[_numpairs*l];
				Real *ch = &_h[_numpoints*l];
				Real   d = DIST(l,l);
				for( i = 0; i < _numpairs; i++ )
					ch[_pairs[2*i]] += ( cy[i] = _wcosts[i] * d );
					//ch[_pairs[2*i]] += ( cy[i] = costfct->computeDistance(i,l,l) ); 
			}

			for( i = 0; i < _numpairs; i++ )
			{
				int id0 = _einfo[i  ].tail;
				int id1 = _einfo[i  ].head;
				int l0  = _pinfo[id0].label;
				int l1  = _pinfo[id1].label;
				
				if ( l0 != l1 )
				{
					Real d  = _wcosts[i]*DIST(l0,l1) - (_y[l0*_numpairs+i] - _y[l1*_numpairs+i] + _wcosts[i]*DIST(l1,l1));
					//Real d  = costfct->computeDistance(i,l0,l1) - (_y[l0*_numpairs+i] - _y[l1*_numpairs+i] + costfct->computeDistance(i,l1,l1));
					UPDATE_BALANCE_VAR0( _y[l0*_numpairs+i], d, _h[l0*_numpoints+id0], _h[l0*_numpoints+id1] );
				}

				_einfo[i].balance = -_y[l1*_numpairs+i]+_wcosts[i]*DIST(l1,l1);
				//_einfo[i].balance = -_y[l1*_numpairs+i]+costfct->computeDistance(i,l1,l1);
			}

			// Get initial primal function
			//
			_APF = 0;
			for( i = 0; i < _numpoints; i++ )
			{
				_pinfo[i].height = _h[_pinfo[i].label*_numpoints+i];
				_APF += _pinfo[i].height;
			}

			//printf( "Done\n" );
		}

		void inner_iteration( Graph::Label label )
		{
			int i;

			Graph       *_graph =  _all_graphs[label];
			Graph::node *_nodes = &_all_nodes [_numpoints*label];
			Graph::arc  *_arcs  = &_all_arcs  [(_numpairs*label)<<1];
			Real        *_cur_y = &_y         [_numpairs*label];

			_time++;
			_graph->flow = 0;

			if ( _APF_change_time < _time - _numlabels )
				return;

			// Update balance and height variables
			//
			Arc_info  *einfo = _einfo;
			Graph::arc *arcs = _arcs;
			Real *cur_y = &_y[_numpairs*label];
			Real *cur_h = &_h[_numpoints*label];
			for( i = 0; i < _numpairs; i++, einfo++, arcs+=2, cur_y++ )
			{
				int l0,l1;
				if ( (l1=_pinfo[einfo->head].label) != label && (l0=_pinfo[einfo->tail].label) != label ) 
				{
					int l = (int) label;
					// Modified by Ahmed
					Real delta  = _wcosts[i]*(DIST(label,l1)+DIST(l0,label)-DIST(l0,l1)-DIST(label,label));
					Real delta1 = _wcosts[i]*DIST(label,l1)-( (*cur_y)+einfo->balance);
					//Real delta  = costfct->computeDistance(i,l,l1)+costfct->computeDistance(i,l0,l)
				        //-costfct->computeDistance(i,l0,l1)-costfct->computeDistance(i,label,label);
					//Real delta1 = costfct->computeDistance(i,l,l1)-( (*cur_y)+einfo->balance);
					Real delta2;
					if ( delta1 < 0 || (delta2=delta-delta1) < 0 ) 
					{
						UPDATE_BALANCE_VAR0( *cur_y, delta1, cur_h[einfo->tail], cur_h[einfo->head] )
						arcs->cap = arcs->r_cap = 0;
#ifndef _METRIC_DISTANCE_
						if ( delta < 0 ) // This may happen only for non-metric distances
						{
							delta = 0;
							_nodes[einfo->head].conflict_time = _time;
						}
#endif
						REV(arcs)->r_cap = delta;
					}
					else
					{
						arcs->cap = arcs->r_cap = delta1;
						REV(arcs)->r_cap = delta2;
					}
				}
				else
				{
					arcs->cap = arcs->r_cap = 0;
					REV(arcs)->r_cap = 0;
				}
			}

			Real total_cap = 0;
			Node_info *pinfo = _pinfo;
			Graph::node *nodes = _nodes;
			for( i = 0; i < _numpoints; i++, pinfo++, nodes++, cur_h++ )
			{
				Real delta = pinfo->height - (*cur_h);
				nodes->tr_cap = delta;
				if (delta > 0) total_cap += delta;
			}
			
			// Run max-flow and update the primal variables
			//
			Graph::flowtype max_flow = _graph -> apply_maxflow(1);
			_APF -= (total_cap - max_flow);
			if ( total_cap > max_flow )
				_APF_change_time = _time;

			cur_y = &_y[_numpairs*label];
			einfo =  _einfo;
			arcs  =  _arcs;
			for( i = 0; i < _numpairs; i++, einfo++, arcs+=2, cur_y++ )
				if ( _pinfo[einfo->head].label != label && _pinfo[einfo->tail].label != label )
				{
					if ( NEW_LABEL(&_nodes[einfo->head]) )
						einfo->balance = -(*cur_y + arcs->cap - arcs->r_cap) +  _wcosts[i]*DIST(label,label);
						//einfo->balance = -(*cur_y + arcs->cap - arcs->r_cap) + costfct->computeDistance(i,label,label);
				}
				else if ( _pinfo[einfo->head].label != label )
				{
					if ( NEW_LABEL(&_nodes[einfo->head]) )
						einfo->balance = -(*cur_y) + _wcosts[i]*DIST(label,label);
						//einfo->balance = -(*cur_y) + costfct->computeDistance(i,label,label);
				}

			cur_h = &_h[_numpoints*label];
			pinfo =  _pinfo;
			nodes =  _nodes;
			for( i = 0; i < _numpoints; i++, pinfo++, nodes++, cur_h++ )
			{
				if ( pinfo->label != label )
				{
					if ( NEW_LABEL(nodes) )
					{
#ifndef _METRIC_DISTANCE_
						if ( nodes->conflict_time > pinfo->time )
						{
							int k;
							for( k = 0; k < pinfo->num_pairs; k++)
							{
								int pid = pinfo->pairs[k];
								if ( pid <= 0 )
								{
									Pair_info *pair = &_pair_info[-pid];
									if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
									{
										int l0 = _pinfo[pair->i0].label;
										int l1 =  pinfo->label;
										// Modified by Ahmed
										Real delta = _wcosts[-pid]*(DIST(l0,label)+DIST(label,l1)-
										                            DIST(l0,l1)-DIST(label,label));
										int pairindex = -pid;
										int l = (int) label;
                                        /*
										Real delta = costfct->computeDistance(pairindex,l0,l)+
											costfct->computeDistance(pairindex,l,l1)-
											costfct->computeDistance(pairindex,l0,l1)-
											costfct->computeDistance(pairindex,label,label)
                                            */
										if ( delta < 0 )
										{
											_cur_y[-pid]  -= delta;
											_einfo[-pid].balance = -_cur_y[-pid] + _wcosts[-pid]*DIST(label,label);
											//_einfo[-pid].balance = -_cur_y[pairindex] + costfct->computeDistance(pairindex,label,label);
											pinfo->height += delta; 
											_APF          += delta;
											_nodes[pair->i0].tr_cap += delta;
										}
									}
								}
							}
						}
#endif

						pinfo->label = label;
						pinfo->height -= nodes->tr_cap;
						nodes->tr_cap = 0;
						pinfo->time = _time;
					}
				}
				*cur_h = pinfo->height;
			}
		}

		void inner_iteration_adapted( Graph::Label label )
		{
			if ( _iter > 1 )
				return track_source_linked_nodes( label );

			int i;
			Graph       *_graph =  _all_graphs[label];
			Graph::node *_nodes = &_all_nodes [_numpoints*label];
			Graph::arc  *_arcs  = &_all_arcs  [(_numpairs*label)<<1];
			Real        *_cur_y = &_y         [_numpairs*label];
			Real        *_cur_h = &_h         [_numpoints*label];

			_time++;
			_graph->flow = 0;

			if ( _APF_change_time < _time - _numlabels )
				return;

			// Update dual vars (i.e. balance and height variables)
			//
			int dt = _time - _numlabels;
			for( i = 0; i < _numpairs; i++ )
			{
				int i0 = _pairs[ i<<1   ];
				int i1 = _pairs[(i<<1)+1];
				if ( _pinfo[i0].time >= dt || _pinfo[i1].time >= dt )
				{
					Graph::arc *arc0 = &_arcs[i<<1];

					if ( _cur_h[i0] != _pinfo[i0].height )
					{
						Real h = _cur_h[i0] - _nodes[i0].tr_cap;
						_nodes[i0].tr_cap = _pinfo[i0].height - h;
						_cur_h[i0] = _pinfo[i0].height;
					}

					if ( _cur_h[i1] != _pinfo[i1].height )
					{
						Real h = _cur_h[i1] - _nodes[i1].tr_cap;
						_nodes[i1].tr_cap = _pinfo[i1].height - h;
						_cur_h[i1] = _pinfo[i1].height;
					}

					int l0,l1;
					if ( (l0=_pinfo[i0].label) != label && (l1=_pinfo[i1].label) != label )
					{
						Graph::arc *arc1 = &_all_arcs[(_numpairs*l1+i)<<1];
						Real y_pq =   _cur_y[i] + arc0->cap - arc0->r_cap ;
						// Modified by Ahmed
						Real y_qp = -(_y[_numpairs*l1+i] + arc1->cap - arc1->r_cap) + _wcosts[i]*DIST(l1,l1);
						Real delta  = _wcosts[i]*(DIST(label,l1)+DIST(l0,label)-DIST(l0,l1)-DIST(label,label));
						Real delta1 = _wcosts[i]*DIST(label,l1)-(y_pq+y_qp);
						int l = (int) label;
						//Real y_qp = -(_y[_numpairs*l1+i] + arc1->cap - arc1->r_cap) + costfct->computeDistance(i,l1,l1);
						//Real delta  = costfct->computeDistance(i,l,l1)+costfct->computeDistance(i,l0,l)
						//	-costfct->computeDistance(i,l0,l1)-costfct->computeDistance(i,label,label);
						//Real delta1 = costfct->computeDistance(i,l,l1)-(y_pq+y_qp);
						Real delta2;
						if ( delta1 < 0 || (delta2=delta-delta1) < 0 ) 
						{
							_cur_y[i] = y_pq+delta1;
							arc0->r_cap = arc0->cap = 0;
#ifndef _METRIC_DISTANCE_
							if ( delta < 0 ) // This may happen only for non-metric distances
							{
								delta = 0;
								_nodes[i1].conflict_time = _time;
							}
#endif
							REV(arc0)->r_cap = delta;

							_nodes[i0].tr_cap -= delta1;
							_nodes[i1].tr_cap += delta1;
						}
						else
						{
							_cur_y[i] = y_pq;
							arc0->r_cap = arc0->cap = delta1;
							REV(arc0)->r_cap = delta2;
						}
					}
					else
					{
						_cur_y[i] += arc0->cap - arc0->r_cap;
						REV(arc0)->r_cap = arc0->r_cap = arc0->cap = 0;	
					}
				}
			}

			// Run max-flow and update the primal variables
			//
			assert( _iter <= 1 );
			Graph::flowtype max_flow = _graph -> apply_maxflow(1);

			double prev_APF = _APF;
			for( i = 0; i < _numpoints; i++ )
			{
				Node_info *pinfo = &_pinfo[i];
				if ( NEW_LABEL(&_nodes[i]) )
				{
#ifndef _METRIC_DISTANCE_
					if ( _nodes[i].conflict_time > pinfo->time )
					{
						Real total_delta = 0;
						int k;
						for( k = 0; k < pinfo->num_pairs; k++)
						{
							int pid = pinfo->pairs[k];
							if ( pid <= 0 )
							{
								Pair_info *pair = &_pair_info[-pid];
								if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
								{
									//Modified by Ahmed
									int l0 = _pinfo[pair->i0].label;
									int l1 =  pinfo->label;
									//int l = (int) label;
									Real delta = _wcosts[-pid]*(DIST(l0,label)+DIST(label,l1)-
									                            DIST(l0,l1)-DIST(label,label));
									//int pairindex = -pid;
									//Real delta = costfct->computeDistance(pairindex,l0,l)+
										// costfct->computeDistance(pairindex,l,l1)-
										// costfct->computeDistance(pairindex,l0,l1)-
										// costfct->computeDistance(pairindex,label,label);
									
									if ( delta < 0 )
									{
										_cur_y[-pid] -= delta;
										total_delta  += delta;
										_nodes[pair->i0].tr_cap += delta;
									}
								}
							}
						}
						if ( total_delta )
							_nodes[i].tr_cap -= total_delta;
					}
#endif

					pinfo->height -= _nodes[i].tr_cap; 
					_APF          -= _nodes[i].tr_cap;
					pinfo->time    = _time;
					pinfo->label   = label;

					if ( pinfo->prev == -2 ) // add to active list
					{
						pinfo->next = _active_list;
						pinfo->prev = -1;
						if (_active_list >= 0)
							_pinfo[_active_list].prev = i;
						_active_list = i;
					}
				}
			}
			if ( _APF < prev_APF )
				_APF_change_time = _time;
		}

		void track_source_linked_nodes( Graph::Label label )
		{
			int i;
			assert( _iter > 1 );

			Graph       *_graph =  _all_graphs[label];
			Graph::node *_nodes = &_all_nodes [_numpoints*label];
			Graph::arc  *_arcs  = &_all_arcs  [(_numpairs*label)<<1];
			Real        *_cur_y = &_y         [_numpairs*label];
			Real        *_cur_h = &_h         [_numpoints*label];

			_time++;
			_graph->flow = 0;

			if ( _APF_change_time < _time - _numlabels )
				return;

			int source_nodes_start1 = -1;
			int source_nodes_start2 = -1;

			int dt = _time - _numlabels;
			i = _active_list;
			while ( i >= 0 )
			{
				Node_info *n = &_pinfo[i];
				int i_next = n->next;

				if ( n->time >= dt )
				{
					if ( _cur_h[i] != n->height )
					{
						Real h = _cur_h[i] - _nodes[i].tr_cap;
						_nodes[i].tr_cap = n->height - h;
						_cur_h[i] = n->height;
					}

					if ( _nodes[i].tr_cap )
					{
                        //if( _nodes[i].tr_cap > 0 )
                        //    std::cout << _nodes[i].tr_cap << std::endl;
                        
						assert(  _nodes[i].tr_cap < 0 );
						_nodes[i].parent  = TERMINAL;
						_nodes[i].is_sink = 1;
						_nodes[i].DIST = 1;
					}
					else _nodes[i].parent = NULL;
				}
				else 
				{
					int prev = n->prev;
					if ( prev >= 0 )
					{
						_pinfo[prev].next = n->next;
						if (n->next >= 0)
							_pinfo[n->next].prev = prev;
					}
					else 
					{
						_active_list = n->next;
						if ( _active_list >= 0 )
							_pinfo[_active_list].prev = -1;
					}
					n->prev = -2;
				}

				i = i_next;
			}

			// Update balance and height variables.
			//
			i = _active_list;
			while ( i >= 0 )
			{
				Node_info *n = &_pinfo[i];
				int i_next = n->next;

				int k;
				Node_info *n0,*n1;
				for( k = 0; k < n->num_pairs; k++)
				{
					int i0,i1,ii;
					Pair_info *pair;
					int pid = n->pairs[k];
					if ( pid >= 0 )
					{
						pair = &_pair_info[pid];
						if ( pair->time == _time )
							continue;

						i0 = i; i1 = pair->i1;
						n0 = n; n1 = &_pinfo[i1];
						ii = i1;
					}
					else
					{
						pid = -pid;
						pair = &_pair_info[pid];
						if ( pair->time == _time )
							continue;

						i1 = i; i0 = pair->i0;
						n1 = n; n0 = &_pinfo[i0];
						ii = i0;
					}
					pair->time = _time;

					int l0,l1;
					Graph::arc *arc0 = &_arcs[pid<<1];
					if ( (l0=n0->label) != label && (l1=n1->label) != label )
					{
						Graph::arc *arc1 = &_all_arcs[(_numpairs*l1+pid)<<1];
						Real y_pq =   _cur_y[pid] + arc0->cap - arc0->r_cap ;
						// Modified by Ahmed
						Real y_qp = -(_y[_numpairs*l1+pid] + arc1->cap - arc1->r_cap) + _wcosts[pid]*DIST(l1,l1);
						//Real y_qp = -(_y[_numpairs*l1+pid] + arc1->cap - arc1->r_cap) + costfct->computeDistance(pid,l1,l1);
						Real delta  = _wcosts[pid]*(DIST(label,l1)+DIST(l0,label)-DIST(l0,l1)-DIST(label,label));
						Real delta1 = _wcosts[pid]*DIST(label,l1)-(y_pq+y_qp);
						int l =(int) label;
						// Real delta  = costfct->computeDistance(pid,l,l1)+costfct->computeDistance(pid,l0,l)
							// -costfct->computeDistance(pid,l0,l1)-costfct->computeDistance(pid,label,label);
						// Real delta1 = costfct->computeDistance(pid,l,l1)-(y_pq+y_qp);
						Real delta2;
						if ( delta1 < 0 || (delta2=delta-delta1) < 0 )
						{
							_cur_y[pid] = y_pq+delta1;
							arc0->r_cap = arc0->cap = 0;
#ifndef _METRIC_DISTANCE_
							if ( delta < 0 ) // This may happen only for non-metric distances
							{
								delta = 0;
								_nodes[i1].conflict_time = _time;
							}
#endif
							REV(arc0)->r_cap = delta;

							_nodes[i0].tr_cap -= delta1;
							_nodes[i1].tr_cap += delta1;

							if ( _pinfo[ii].prev == -2 && _source_nodes_tmp2[ii] == -2 )
							{
								_source_nodes_tmp2[ii] = source_nodes_start2;
								source_nodes_start2 = ii;
							}
						}
						else
						{
							_cur_y[pid] = y_pq;
							arc0->r_cap = arc0->cap = delta1;
							REV(arc0)->r_cap = delta2;
						}
					}
					else
					{
						_cur_y[pid] += arc0->cap - arc0->r_cap;
						REV(arc0)->r_cap = arc0->r_cap = arc0->cap = 0;	
					}
				}

				Graph::node *nd = &_nodes[i];
				if ( nd->tr_cap > 0 )
				{
					nd -> is_sink = 0;
					nd -> parent = TERMINAL;
					nd -> DIST = 1;

					_graph->set_active(nd);
					
					_source_nodes_tmp1[i] = source_nodes_start1;
					source_nodes_start1 = i;
				}
				else if (nd->tr_cap < 0)
				{
					nd -> is_sink = 1;
					nd -> parent = TERMINAL;
					nd -> DIST = 1;
				}
				else nd -> parent = NULL;
				//n -> TS = 0;

				i = i_next;
			}

			for( i = source_nodes_start2; i >= 0; )
			{
				Graph::node *nd = &_nodes[i];
				if ( nd->tr_cap > 0 )
				{
					nd -> is_sink = 0;
					nd -> parent = TERMINAL;
					nd -> DIST = 1;

					_graph->set_active(nd);
					
					_source_nodes_tmp1[i] = source_nodes_start1;
					source_nodes_start1 = i;
				}
				else if (nd->tr_cap < 0)
				{
					nd -> is_sink = 1;
					nd -> parent = TERMINAL;
					nd -> DIST = 1;
				}
				else nd -> parent = NULL;
				//n -> TS = 0;
				
				int tmp = i;
				i = _source_nodes_tmp2[i];
				_source_nodes_tmp2[tmp] = -2;
			}

			// Run max-flow and update primal variables
			//
			Graph::flowtype max_flow = _graph -> apply_maxflow(0);

			double prev_APF = _APF;
			int num_children = 0;
			for( i = source_nodes_start1; i >= 0; )
			{
				Graph::node *n = &_nodes[i];
				if ( n->parent == TERMINAL )
				{
					Node_info *pinfo = &_pinfo[i];

#ifndef _METRIC_DISTANCE_
					if ( n->conflict_time > pinfo->time )
					{
						Real total_delta = 0;
						int k;
						for( k = 0; k < pinfo->num_pairs; k++)
						{
							int pid = pinfo->pairs[k];
							if ( pid <= 0 )
							{
								Pair_info *pair = &_pair_info[-pid];
								if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
								{
									int l0 = _pinfo[pair->i0].label;
									int l1 =  pinfo->label;
									// Modified by Ahmed
									Real delta = _wcosts[-pid]*(DIST(l0,label)+DIST(label,l1)-
									                           DIST(l0,l1)-DIST(label,label));
									int pairindex = -pid;
									// Real delta = costfct->computeDistance(pairindex,l0,label)+
										// costfct->computeDistance(pairindex,label,l1)-
										// costfct->computeDistance(pairindex,l0,l1)-
										// costfct->computeDistance(pairindex,label,label);
									if ( delta < 0 )
									{
										_cur_y[-pid] -= delta;
										total_delta  += delta;
										_nodes[pair->i0].tr_cap += delta;
									}
								}
							}
						}
						if ( total_delta )
							n->tr_cap -= total_delta;
					}
#endif

					pinfo->height   -= n->tr_cap; 
					_APF            -= n->tr_cap;
					pinfo->label     = label;
					pinfo->time      =_time;

					if ( pinfo->prev == -2 ) // add to active list
					{
						pinfo->next = _active_list;
						pinfo->prev = -1;
						if (_active_list >= 0)
							_pinfo[_active_list].prev = i;
						_active_list = i;
					}

					Graph::arc *a;
					for ( a=n->first; a; a=a->next )
					{
						Graph::node *ch = a->head;
						if ( ch->parent == a->sister )
							_children[num_children++] = ch;
					}
				}

				int tmp = i;
				i = _source_nodes_tmp1[i];
				_source_nodes_tmp1[tmp] = -2;
			}

			for( i = 0; i < num_children; i++ )
			{
				Graph::node *n = _children[i];
				//unsigned int id  = ((unsigned int)n - (unsigned int)_nodes) / sizeof(Graph::node); 
				unsigned int id  = (unsigned int)(long long int)((n - _nodes) / sizeof(Graph::node)); 
				Node_info *pinfo = &_pinfo[id];

#ifndef _METRIC_DISTANCE_
				if ( n->conflict_time > pinfo->time )
				{
					Real total_delta = 0;
					int k;
					for( k = 0; k < pinfo->num_pairs; k++)
					{
						int pid = pinfo->pairs[k];
						if ( pid <= 0 )
						{
							Pair_info *pair = &_pair_info[-pid];
							if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
							{
								Graph::Label l0 = _pinfo[pair->i0].label;
								Graph::Label l1 =  pinfo->label;
								// Modified by Ahmed
								Real delta = _wcosts[-pid]*(DIST(l0,label)+DIST(label,l1)-
								                           DIST(l0,l1)-DIST(label,label));
								int pairindex = -pid;
								// Real delta = costfct->computeDistance(pairindex,l0,label)+
									// costfct->computeDistance(pairindex,label,l1)-
									// costfct->computeDistance(pairindex,l0,l1)-
									// costfct->computeDistance(pairindex,label,label);

								if ( delta < 0 )
								{
									_cur_y[-pid] -= delta;
									total_delta  += delta;
									_nodes[pair->i0].tr_cap += delta;
								}
							}
						}
					}
					if ( total_delta )
						n->tr_cap -= total_delta;
				}
#endif

				pinfo->height   -= n->tr_cap; 
				_APF            -= n->tr_cap;
				pinfo->label     = label;
				pinfo->time      =_time;

				if ( pinfo->prev == -2 ) // add to active list
				{
					pinfo->next = _active_list;
					pinfo->prev = -1;
					if (_active_list >= 0)
						_pinfo[_active_list].prev = id;
					_active_list = id;
				}

				Graph::arc *a;
				for ( a=n->first; a; a=a->next )
				{
					Graph::node *ch = a->head;
					if ( ch->parent == a->sister )
						_children[num_children++] = ch;
				}
			}

			if ( _APF < prev_APF )
				_APF_change_time = _time;
		}

        void run( void )
        {
			double total_t = 0, total_augm = 0;
			init_duals_primals();

			int iter = 0;
			while ( iter < _max_iters )
			{
				double prev_APF = _APF;
				_iter = iter;

			//	printf( "Iter %d: ", iter );
				clock_t start = clock();
				if ( !iter )
				{
					for( Graph::Label l = 0; l < _numlabels; l++ )
						inner_iteration( l );
				}
				else 
				{
					for( Graph::Label l = 0; l < _numlabels; l++ )
						inner_iteration_adapted( l );
				}
				clock_t finish = clock();
				float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
				total_t += t;
				//printf( "APF = %.0f (%.3f secs)\n", _APF, t );

				if ( prev_APF <= _APF )
					break;

				iter++;
			}
			//printf( "Total time = %f\n", total_t );
			//printf("%d iterations / ",iter);
        }

		void fillGraph( Graph *_graph ) 
		{
			_graph->add_nodes();
			_graph->add_edges( _pairs, _numpairs );
		}

		void createNeighbors( void )
		{
			// Fill auxiliary structures related to neighbors
			//
			int i;
			_pairs_arr = new int[_numpairs*2];

			for( i = 0; i < _numpoints; i++ )
				_pinfo[i].num_pairs = 0;

			for( i = 0; i < _numpairs; i++ )
			{
				int i0 = _pairs[i<<1];
				int i1 = _pairs[(i<<1)+1];
				_pinfo[i0].num_pairs++; 
				_pinfo[i1].num_pairs++;
			}

			int offset = 0;
			for( i = 0; i < _numpoints; i++ )
			{
				_pinfo[i].pairs = &_pairs_arr[offset];  
				offset += _pinfo[i].num_pairs;
				_pinfo[i].num_pairs = 0;
			}

			_pair_info = new Pair_info[_numpairs];
			_einfo = new Arc_info[_numpairs];

			for( i = 0; i < _numpairs; i++ )
			{
				int i0 = _pairs[i<<1];
				int i1 = _pairs[(i<<1)+1];
				_pinfo[i0].pairs[_pinfo[i0].num_pairs++] =  i;
				_pinfo[i1].pairs[_pinfo[i1].num_pairs++] = -i;

				_einfo[i].tail = i0; 
				_einfo[i].head = i1;

				_pair_info[i].i0 = i0; 
				_pair_info[i].i1 = i1;
				_pair_info[i].time = -1;
			}
		}

		static void err_fun(char * msg)
		{
			printf("%s",msg);
		}

		struct Node_info
		{
			Graph::Label label; 
			Real height; 
			TIME time;
 			int next;    
			int prev;
			int *pairs; 
			int num_pairs;
		};
		
        int           _numpoints;
        int           _numpairs;
        int           _numlabels;
		int           _max_iters;
        Real         *_h; // height variables
		Real         *_y; // balance variables
		Real         *_dist; // distance function for pairwise potential
        int          *_pairs;
		Graph::node  *_all_nodes; 
		Graph::arc   *_all_arcs;  
		Graph       **_all_graphs;
		int          *_source_nodes_tmp1; 
		int          *_source_nodes_tmp2; 
		Real         *_wcosts; 
		int           _iter;
		Node_info    *_pinfo; 
		int         *_primal0;
		//CostFunction *costfct;

    private:

		// auxiliary data structures and variables
		//
		struct Pair_info
		{
			int i0, i1;
			TIME time;
		};

		struct Arc_info
		{
			int head, tail;
			Real balance;
		};

		Arc_info     *_einfo;
		double        _APF;
		int           _time;
		Pair_info    *_pair_info;
		int           _active_list;
		int           _APF_change_time;
		Graph::node **_children;
		int          *_pairs_arr;
	
        // Assignment or copying are not allowed
        //
        CV_Fast_PD( const CV_Fast_PD &other );
        CV_Fast_PD operator=( const CV_Fast_PD &other );
};

#endif /* __FAST_PD_H__ */

//#############################################################################
//#
//# EOF
//#
//#############################################################################

