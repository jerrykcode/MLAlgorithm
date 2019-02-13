#include "DecisionTree.h"
#include "DecisionTree.cpp"

int main() {
#ifdef DEBUG
	DecisionTree<double> dtree;
	int nPoints, nAttributes, nClusters;
	cin >> nPoints >> nAttributes >> nClusters;
	double *dataBuffer = new double[nPoints * nAttributes];
	int *label = new int[nPoints];
	int *nChildren = new int[nAttributes];
	for (int i = 0; i < nPoints * nAttributes; i++)
		cin >> dataBuffer[i];
	for (int i = 0; i < nPoints; i++)
		cin >> label[i];
	for (int i = 0; i < nAttributes; i++)
		cin >> nChildren[i];
	dtree.train(dataBuffer, nPoints, nAttributes, label, nChildren, nClusters);
	int k;
	cin >> k;
	double *predict_point = new double[nAttributes];
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < nAttributes; j++)
			cin >> predict_point[j];
		cout << dtree.predict(predict_point) << endl;
	}
#endif
	return 0;
}
