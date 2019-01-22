#include <iostream>
#include <vector>
#include <map>
using namespace std;

#define KNN_ERROR_FLAG -1
#define KNN_DIMENSION_ERROR (KNN_ERROR_FLAG)
#define KNN_NODATA_ERROR (KNN_ERROR_FLAG - 1)

class Knn {
public:
	
	Knn();
	~Knn();
	
	void train(double *dataBuffer, int nPoints/*nRows*/, int dimension/*nCols*/, int *label);
	int predict(int k, double *pointBuffer, int pointBuffer_size);

private:

	typedef struct Point {
		int id;
		double *data;
		
		Point(int id, double *data) : id(id), data(data) {}	
	} *PPoint;

	typedef struct Neighbor {
		double distance;
		PPoint p_neighborPoint;
		Neighbor(double distance, PPoint p_neighborPoint) : distance(distance), p_neighborPoint(p_neighborPoint) {}
	} *PNeighbor;

	class MinHeap  {
	public:
		MinHeap(int capacity);
		~MinHeap();
		void clearHeap();
		void insert(PNeighbor pNeighbor);
		void top(PNeighbor pNeighbor);
	private:
		int size_;
		int capacity_;
		PNeighbor *elements_;
	};

	int k_;
	int nPoints_;
	int dimension_;
	vector<PPoint> dataSet_;
	bool hasDataLoaded_;
	int *label_;
	PPoint p_predictPoint_;

	MinHeap *heap_;
	void loadBuffer(double *dataBuffer);
	void clearData();
	void calc_distance();
	int mostFrequencyLabel();
};
