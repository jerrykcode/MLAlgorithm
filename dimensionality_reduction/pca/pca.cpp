#include "pca.h"

PCA::PCA() {

}

PCA::~PCA() {

}

PCA::Matrix::Matrix(int nRows, int nCols) : nRows(nRows), nCols(nCols) {
	if (nRows <= 0 || nCols <= 0) return;
	data = new double*[nRows];
	for (int i = 0; i < nRows; i++) {
		data[i] = new double[nCols];
	}
}

PCA::Eigen::Eigen(double eigenvalue, double *eigenvector) : eigenvalue(eigenvalue), eigenvector(eigenvector) {
	
}

void PCA::transform(int dimension, double *dataBuffer, int nPoints, int originDimension, double *outputBuffer) {
	int nRows = originDimension;
	int nCols = nPoints;
	PMatrix pMatrix = new Matrix(nRows, nCols);
	loadBuffer(dataBuffer, pMatrix);
	mean_matrix(pMatrix);
	PMatrix pCovMatrix = new Matrix(nRows, nRows);
	calc_covMatrix(pMatrix, pCovMatrix);
	vector<Eigen> eigens;
	calc_eigen(pCovMatrix, eigens, 0.0001, pCovMatrix->nRows * pCovMatrix->nRows * 30);
	clear_matrix(pCovMatrix);
	delete pCovMatrix;
	pCovMatrix = NULL;
	calc_reducedMatrix(eigens, dimension, pMatrix, outputBuffer);
	clear_matrix(pMatrix);
	delete pMatrix;
	pMatrix = NULL;
	for (auto it = eigens.begin(); it != eigens.end(); it++) {
		free(it->eigenvector);
	}
	eigens.clear();
	vector<Eigen>().swap(eigens);
}

void PCA::loadBuffer(double *dataBuffer, PMatrix pMatrix) {
	for (int i = 0; i < pMatrix->nRows; i++)
		for (int j = 0; j < pMatrix->nCols; j++) {
			pMatrix->data[i][j] = *(dataBuffer + j * pMatrix->nRows + i);
		}
}

void PCA::mean_matrix(PMatrix pMatrix) {
	for (int i = 0; i < pMatrix->nRows; i++) {
		double row_average = 0;
		for (int j = 0; j < pMatrix->nCols; j++)
			row_average += pMatrix->data[i][j] / pMatrix->nCols;
		for (int j = 0; j < pMatrix->nCols; j++)
			pMatrix->data[i][j] -= row_average;
	}
}

void PCA::calc_covMatrix(PMatrix pMatrix, PMatrix pCovMatrix) {
	if (pCovMatrix->nRows != pMatrix->nRows || pCovMatrix->nCols != pMatrix->nRows) return;
	for (int i = 0; i < pCovMatrix->nRows; i++)
		for (int j = i; j < pCovMatrix->nCols; j++) {
			pCovMatrix->data[i][j] = 0;
			for (int k = 0; k < pMatrix->nCols; k++)
				pCovMatrix->data[i][j] += (pMatrix->data[i][k] * pMatrix->data[j][k]) / pMatrix->nCols;
		}
	for (int i = 0; i < pCovMatrix->nRows; i++)
		for (int j = 0; j < i; j++)
			pCovMatrix->data[i][j] = pCovMatrix->data[j][i];
}

void PCA::calc_eigen(PMatrix pMatrix, vector<Eigen>& eigens, double eps, int maxIter) {
	if (pMatrix->nRows != pMatrix->nCols) {
		return;
	}
	int iterCount = 0;
	double **eigenvectors = new double*[pMatrix->nRows];
	for (int i = 0; i < pMatrix->nRows; i++) {
		eigenvectors[i] = new double[pMatrix->nCols];
		fill(eigenvectors[i], eigenvectors[i] + pMatrix->nCols, 0);
		eigenvectors[i][i] = 1;
	}
	while ((iterCount++) < maxIter) {
		double maxValue = 0;
		int maxRow, maxCol;
		double **data = pMatrix->data;
		for (int i = 0; i < pMatrix->nRows; i++)
			for (int j = i + 1; j < pMatrix->nCols; j++) {
				double val = abs(data[i][j]);
				if (val > maxValue) {
					maxValue = val;
					maxRow = i;
					maxCol = j;
				}
			}
		if (maxValue < eps) break;
		double aii = data[maxRow][maxRow];
		double ajj = data[maxCol][maxCol];
		double aij = data[maxRow][maxCol]; //aji = aij
		double w = (ajj - aii) / (2 * aij);
		double t; //tan
		if (w > 0)
			t = sqrt(w * w + 1) - w;
		else 
			t = -1 * sqrt(w * w + 1) - w;
		double c = 1 / sqrt(1 + t * t); //cos
		double s = c * t; //sin

		//update matrix
		data[maxRow][maxRow] = c * c * aii - 2 * s * c * aij + s * s * ajj;
		data[maxCol][maxCol] = s * s * aii + 2 * s * c * aij + c * c * ajj;
		data[maxRow][maxCol] = s * c * (aii - ajj) + (c * c - s * s)*aij;
		data[maxCol][maxRow] = data[maxRow][maxCol];
		
		for (int k = 0; k < pMatrix->nCols; k++) {
			if (k != maxRow && k != maxCol) {
				double aik = data[maxRow][k];
				double ajk = data[maxCol][k];
				data[maxRow][k] = c * aik - s * ajk;
				data[maxCol][k] = c * ajk + s * aik;
			}
		}	
		for (int k = 0; k < pMatrix->nRows; k++) {
			if (k != maxRow && k != maxCol) {
				double aki = data[k][maxRow];
				double akj = data[k][maxCol];
				data[k][maxRow] = c * aki - s * akj;
				data[k][maxCol] = c * akj + s * aki;
			}
		}
		//update eigen vector matrix
		for (int k = 0; k < pMatrix->nRows; k++) {
			double eki = eigenvectors[k][maxRow];
			double ekj = eigenvectors[k][maxCol];
			eigenvectors[k][maxRow] = c * eki + s * ekj;
			eigenvectors[k][maxCol] = c * ekj - s * eki;
		}
	} //while
	for (int i = 0; i < pMatrix->nRows; i++) {		
		double m = 0;
		for (int j = 0; j < pMatrix->nCols; j++)
			m += eigenvectors[i][j] * eigenvectors[i][j];
		m = sqrt(m);
		for (int j = 0; j < pMatrix->nCols; j++)
			eigenvectors[i][j] /= m;
		eigens.push_back(Eigen(pMatrix->data[i][i], eigenvectors[i]));
	}	
	free(eigenvectors);
}

#ifdef TEST_EIGEN
	void PCA::test_eigen() {
		cout << "Test PCA::calc_eigen function" << endl;
		cout << "Enter i for input matrix, d for default matrix:" << endl;
		char cmd;
		cin >> cmd;
		int nRows;
		PMatrix pMatrix = NULL;
		if (cmd == 'i') {
			cout << "Enter the number of rows(cols):" << endl;
			cin >> nRows;
			pMatrix = new Matrix(nRows, nRows);
			for (int i = 0; i < nRows; i++)
				for (int j = 0; j < nRows; j++) {
					cout << "Enter the element of row" << i << "and col" << j << endl;
					cin >> pMatrix->data[i][j];
				}			
		}
		else if (cmd == 'd') {
			nRows = 2;
			pMatrix = new Matrix(nRows, nRows);
			pMatrix->data[0][0] = 3;
			pMatrix->data[0][1] = 1;
			pMatrix->data[1][0] = 1;
			pMatrix->data[1][1] = 3;
		}
		else return;
		vector<Eigen> eigens;
		calc_eigen(pMatrix, eigens, 0.0001, pMatrix->nRows * pMatrix->nRows * 30);
		for (auto it = eigens.begin(); it != eigens.end(); it++) {
			cout << "eigen value : " << it->eigenvalue << "; eigen vector :";
			for (int i = 0; i < nRows; i++) {
				putchar(' ');
				cout << it->eigenvector[i];
			}
			cout << endl;
		}
		for (int i = 0; i < nRows; i++)
			free(pMatrix->data[i]);
		free(pMatrix->data);
		for (auto it = eigens.begin(); it != eigens.end(); it++) 
			free(it->eigenvector);
		eigens.clear();
		vector<Eigen>().swap(eigens);
	}
#endif

bool compareEigen(PCA::Eigen a, PCA::Eigen b) {
	return a.eigenvalue > b.eigenvalue;
}

void PCA::calc_reducedMatrix(vector<Eigen>& eigens, int dimension, PMatrix pMatrix, double *outputBuffer) {
	sort(eigens.begin(), eigens.end(), compareEigen);
	for (int i = 0; i < dimension; i++) {
		double *ev = eigens[i].eigenvector;
		for (int j = 0; j < pMatrix->nCols; j++) {
			double result_ij = 0; //(i, j) element of the result matrix
			for (int k = 0; k < pMatrix->nRows; k++) {
				result_ij += ev[k] * pMatrix->data[k][j];
			}
			//(i, j) element of the result matrix is the (j, i) element of the output matrix
			*(outputBuffer + j * dimension + i) = result_ij;
		}
	}
}

void PCA::clear_matrix(PMatrix pMatrix) {
	for (int i = 0; i < pMatrix->nRows; i++)
		free(pMatrix->data[i]);
	free(pMatrix->data);
}
