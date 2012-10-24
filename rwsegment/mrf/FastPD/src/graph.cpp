/* graph.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001. */

#include <stdio.h>
#include "graph.h"

Graph::Graph(node *nodes, arc *arcs, int num_nodes, void (*err_function)(char *))
{
	error_function = err_function;
	_nodes = nodes;
	_arcs  = arcs ;
	_num_nodes = num_nodes;
	flow = 0;
}

Graph::~Graph()
{
}

void Graph::add_nodes()
{
	for( int i = 0; i < _num_nodes; i++ )
	{
		_nodes[i].first = NULL;
		_nodes[i].tr_cap = 0;
#ifndef _METRIC_DISTANCE_
		_nodes[i].conflict_time = -1;
#endif
	}
}

void Graph::add_edges(int *pairs, int numpairs )
{
	captype cap = 0;
	captype rev_cap = 0;
	for( int i = 0; i < 2*numpairs; i+=2 )
	{
		node_id from = &_nodes[pairs[i]];
		node_id to   = &_nodes[pairs[i+1]];

		arc *a, *a_rev;

		a = &_arcs[i];
		a_rev = a + 1;

		a -> sister = a_rev;
		a_rev -> sister = a;
		a -> next = ((node*)from) -> first;
		((node*)from) -> first = a;
		a_rev -> next = ((node*)to) -> first;
		((node*)to) -> first = a_rev;
		a -> head = (node*)to;
		a_rev -> head = (node*)from;
		a -> r_cap = cap;
		a_rev -> r_cap = rev_cap;
	}
}

void Graph::set_tweights(node_id i, captype cap_source, captype cap_sink)
{
	flow += (cap_source < cap_sink) ? cap_source : cap_sink;
	((node*)i) -> tr_cap = cap_source - cap_sink;
}

void Graph::add_tweights(node_id i, captype cap_source, captype cap_sink)
{
	register captype delta = ((node*)i) -> tr_cap;
	if (delta > 0) cap_source += delta;
	else           cap_sink   -= delta;
	flow += (cap_source < cap_sink) ? cap_source : cap_sink;
	((node*)i) -> tr_cap = cap_source - cap_sink;
}

