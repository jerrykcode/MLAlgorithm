#include <iostream>
#include <string>
#include <vector>
#include <queue>
using namespace std;

class Dbscan {
public:
	Dbscan();
	~Dbscan();

	int DBSCAN(double dist, int minPts, double *dataBuffer, int nPoints/*nRows*/, int dimension/*nCols*/, int *label);
	
	int isNoiseLabel(int label);
private:
	
typedef struct Point {
	int id;
	double *data;
	bool isCorePoint;
	vector<int> neighborPts; //id of the neighbor points
	Point(int id, double *data) : id(id), data(data), isCorePoint(false) {} 
} *PPoint;

	int nPoints;
	int dimension;
	double dist;
	int minPts;
	vector<PPoint> dataSet;
	vector<PPoint> corePoints;

	void loadBuffer(double *dataBuffer);
	int db_clustering(int *label);
	void findNeighborPoints();
	void findCorePoints();
	int connectCorePoints(int *label);
	void clear();
};
