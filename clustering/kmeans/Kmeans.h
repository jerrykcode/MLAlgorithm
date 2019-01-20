#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <map>
using namespace std;

class Kmeans {
public:
	Kmeans(); 
	~Kmeans();

	void clustering(int nClusters, double *dataBuffer, int nRows, int nCols, int *label);
private:

typedef struct Point {	
	double *data;
	int dimension;
	Point(double *data, int dimension) : data(data), dimension(dimension) {}
} *PPoint;

	int nClusters;
	int nRows;
	int nCols;
	vector<Point> dataSet;
	vector<Point> centroids;

	void loadBuffer(double *dataBuffer, int nRows, int nCols);
	void kmeans_clustering(int *label);
	void randomSetCentroids();
	void clearData();
};
