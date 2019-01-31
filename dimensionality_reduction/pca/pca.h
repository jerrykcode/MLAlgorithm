#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
using namespace std;

class PCA {
public:
	PCA();
	~PCA();

	void transform(int dimension, double *dataBuffer, int nPoints, int originDimension, double *outputBuffer);

#ifdef TEST_EIGEN
	void test_eigen();	
#endif

	typedef struct Eigen {
		double eigenvalue;
		double *eigenvector;
		Eigen(double eigenvalue, double *eigenvector);
	} *PEigen;

private:
	typedef struct Matrix {
		int nRows;
		int nCols;
		double **data;
		Matrix(int nRows, int nCols);
	} *PMatrix;	

	void loadBuffer(double *dataBuffer, PMatrix pMatrix);
	void mean_matrix(PMatrix pMatrix);
	void calc_covMatrix(PMatrix pMatrix, PMatrix pCovMatrix);	
	void calc_eigen(PMatrix pMatrix, vector<Eigen>& eigens, double eps, int maxIter);
	void calc_reducedMatrix(vector<Eigen>& eigens, int dimension, PMatrix pMatrix, double *outputBuffer);
	void clear_matrix(PMatrix pMatrix);
};
