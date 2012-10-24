#ifndef __COSTFUNCTION
#define __COSTFUNCTION





typedef double Real;

class CostFunction
{
public :
	CostFunction(){}
	CostFunction(Real * sigma, Real * mean, Real & maxDisplacement, 
		int & numberOfLabels, int * pairs, Real * position, int & numberOfPoints,
		int & numberOfPairs,Real * Dx,Real * Dy,Real _norm,Real _alpha);
	~CostFunction();
	Real computeDistance(int & pair, int & label1, int & label2);
	Real * _sigma;
	Real * _mean;
	int       * _pairs;
	Real     * _position;
	Real       _maxDisplacement;
	int         _numberOfLabels;
	int         _numberOfPoints;
	int         _numberOfPairs;
	Real       _normalize;
	Real minSigma;
	Real * dx;
	Real * dy;
	Real alpha;
};

#endif