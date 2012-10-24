//#############################################################################
//#
//# PD_wrapper.cpp:
//#  Matlab mex file
//#
//#  @author Nikos Komodakis
//#  
//#############################################################################

#include <stdlib.h>
#include "mex.h"
#include "CV_Fast_PD.h"
#include "CV_Matlab.h"

//typedef CV_Fast_PD::Real Real;
typedef double Real;

//extern "C" _export void mexFunction( int nlhs, mxArray *plhs[],
//                                     int nrhs, const mxArray *prhs[] );
//extern "C"  __declspec( dllexport ) void mexFunction( int nlhs, mxArray *plhs[],
//                                     int nrhs, const mxArray *prhs[] );
CV_Matlab::Matrix<double> lcosts;


// The next 2 matrices contain indices that follow 
// the C-indexing scheme i.e the smallest index is 0
//
CV_Matlab::Matrix<double> pairs;   
CV_Matlab::Matrix<double> primal0;

void mexFunction( int nlhs, mxArray *plhs[],
				 int nrhs, const mxArray *prhs[])
{
	// Check number of input & output arguments
	//
	char msg [100];
	int max_arg=11;
	sprintf(msg,"%d or %d inputs required",max_arg,max_arg+1);
	if ( nrhs != max_arg && nrhs != max_arg+1)
		mexErrMsgTxt(msg);
	int min_out=1;
	sprintf(msg,"%d minimum outputs required",min_out);
	if ( nlhs < min_out)
		mexErrMsgTxt(msg);

	// Get input arguments
	//
	int id = 0;
	
	//arg 1
	lcosts.reset( (mxArray *)prhs[id++] );
	//arg 2
	pairs.reset ( (mxArray *)prhs[id++] );
	//arg 3
	Real * sigma   = (Real *) mxGetData((mxArray *)prhs[id++]);
	//arg 4
	Real * mean    = (Real *) mxGetData((mxArray *)prhs[id++]);
	//arg 5
	Real * position= (Real *) mxGetData((mxArray *)prhs[id++]);
	//arg 6
	Real _maxD    = (Real)( mxGetPr(prhs[id++]) )[0];
	////arg 7
	//int _maxLevel    = (int)( mxGetPr(prhs[id++]) )[0];

	//arg 7
	int _max_iters = (int)( mxGetPr(prhs[id++]) )[0];
	if ( _max_iters <= 0 ) 
		_max_iters = 1;

	//arg 8
	Real * dx   = (Real *) mxGetData((mxArray *)prhs[id++]);

	//arg 9
	Real * dy   = (Real *) mxGetData((mxArray *)prhs[id++]);

	//arg 10
	Real _norm    = (Real)( mxGetPr(prhs[id++]) )[0];

	//arg 11
	Real alpha    = (Real)( mxGetPr(prhs[id++]) )[0];
	
	int primal0_on;
	if ( nrhs > max_arg )
	{
		primal0.reset( (mxArray *)prhs[id++] );
		primal0_on = 1;
	}
	else primal0_on = 0;

	// Prepare input for C++ routine
	//
	int _numpoints = lcosts.height();
	int _numlabels = lcosts.width();
	int _numpairs  = pairs.height();
	
	if ( primal0_on )
		if ( primal0.width()*primal0.height() != _numpoints )
			mexErrMsgTxt( "primal0 vector has illegal size" );

	Real *_lcosts;
	int  *_pairs ;
	int *_primal0 = NULL;


	if ( primal0_on )
	{
		_primal0 = new int[_numpoints];
		for( int i = 0; i < _numpoints; i++ )
			_primal0[i] = (primal0.data())[i];
	}

	_lcosts = new Real [_numpoints* _numlabels];
	for( int i = 0; i < _numpoints; i++ )
	{
		for( int j = 0; j < _numlabels; j++ )
			_lcosts[i+_numpoints*j] = lcosts(j,i);
	}

	_pairs = new int [2*_numpairs];
	for( int i = 0; i < _numpairs; i++ )
	{
		for( int j = 0; j < 2; j++ )
			_pairs[i*2+j] = pairs(j,i)-1;
	}
	
	
	// Run algorithm
	//
	/*Real *prim = new Real[_numpoints];*/
	CostFunction cost(sigma,mean,_maxD,
		_numlabels, _pairs, position,
		_numpoints, _numpairs,dx,dy,_norm,alpha);

	

	/*for(int i=0;i<_maxLevel;i++)
	{*/
		CV_Fast_PD em( _numpoints, _numlabels, _lcosts,
						   _numpairs, _pairs, & cost,
						   _max_iters, _primal0); 
		em.run();
		//for( int j = 0; j < _numpoints; j++ )
		//{
		//	prim[j] = em->_pinfo[j].label;     
		//}
		//delete em;
		//double third=1.0/3;
		//int NOL=(int) (pow(_numlabels,third)+0.5);
		///*_maxD/=(NOL-1);
		//_maxD*=2;*/
		//cost.update(prim/*,_maxD*/);
		//cost.update_norm();
		//for( int n = 0; n < _numpoints; n++ )
		//{
		//	for( int j = 0; j < _numlabels; j++ )
		//		_lcosts[n+_numpoints*j] = lcosts(j,n);
		//}
	/*}*/
	
	// Return results
	//
	plhs[0] = mxCreateDoubleMatrix( _numpoints, 1, mxREAL );
	CV_Matlab::Matrix<double> primal( plhs[0] ); // Contains indices that
	for( int i = 0; i < _numpoints; i++ )        // follow the C-indexing
		primal(0,i) = em._pinfo[i].label;        // scheme
		//primal(0,i) = prim[i];        // scheme
	Real energy=0;
	/*for(int i=0;i<_numpoints;i++)
	{
		energy+=lcosts(primal(0,i),i);
	}*/
	for(int i=0;i<_numpairs;i++)
	{
		int l1=primal(0,_pairs[i*2]);
		int l2=primal(0,_pairs[i*2+1]);
		energy+=cost.computeDistance(i,l1,l2);
	}
	mexPrintf("Energy = %f\n",energy);
	mexEvalString("pause(0.01);");
	plhs[1] = mxCreateDoubleMatrix( 1, 1, mxREAL );
	CV_Matlab::Matrix<double> NRJ( plhs[1] );
	NRJ(0,0)=energy;
	// Release memory
	//
	if ( _primal0 )
		delete [] _primal0;

	
	delete [] _lcosts;

	delete [] _pairs;
	//delete [] prim;
}

