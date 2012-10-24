#include "CostFunction.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include "mex.h"
using namespace std;



CostFunction::CostFunction(Real * sigma, Real * mean, Real & maxDisplacement, 
		int & numberOfLabels, int * pairs, Real * position, int & numberOfPoints,
		int & numberOfPairs,Real * Dx,Real * Dy,Real _norm,Real _alpha)
{
	dx=Dx;
	dy=Dy;
	_normalize=_norm,
	_numberOfLabels =numberOfLabels;
	_pairs          =pairs;
	_position       =position;
	_numberOfPoints =numberOfPoints;
	_numberOfPairs  =numberOfPairs;

	_sigma          = new Real[_numberOfPairs];
	_mean           = new Real[_numberOfPairs];
	minSigma=100000000000;
	for(int i=0;i<_numberOfPairs;i++)
	{
		_sigma[i]=sigma[i];
		if(_sigma[i]<minSigma)
			minSigma=_sigma[i];
		_mean [i]=mean [i];
	}
	alpha=_alpha;
}


CostFunction::~CostFunction()
{
	if (_sigma)
		delete [] _sigma;
	if (_mean)
		delete [] _mean;
}


Real CostFunction::computeDistance(int & pair, int & label1, int & label2)
{
	Real constant=0.5*log(minSigma*2*3.14159265);
	int p1=_pairs[2*pair];
	int p2=_pairs[2*pair+1];
	Real temp[2]={0,0};
	temp[0]=_position[p2]-_position[p1]+dx[label2]-dx[label1];
	temp[1]=_position[p2+_numberOfPoints]-_position[p1+_numberOfPoints]+dy[label2]-dy[label1];
	Real val=sqrt(temp[0]*temp[0]+temp[1]*temp[1])/_normalize;
	Real distance = (0.5*log(_sigma[pair]*2*3.14159265)+(val-_mean[pair])*(val-_mean[pair])/
		(2*_sigma[pair])-constant)*_numberOfPoints/_numberOfPairs;
	if (distance <0)
	{
		mexPrintf("distance négative pair %d labels %d %d \n",pair,label1,label2);
		mexEvalString("pause;");
	}
	//mexPrintf("distance pair %d labels %d %d = %f\n",pair,label1,label2,distance);
	//mexEvalString("pause(0.01);");
	return  alpha*distance;
}