#include "Kmeans.h"

Kmeans::Kmeans() {

}

Kmeans::~Kmeans() {
	vector<Point>().swap(dataSet);
	vector<Point>().swap(centroids);
}

void Kmeans::clustering(int nClusters, double *dataBuffer, int nRows, int nCols, int *label) {
	this->nClusters = nClusters;
	this->nRows = nRows;
	this->nCols = nCols;
	loadBuffer(dataBuffer, nRows, nCols);	
	kmeans_clustering(label);
	clearData();	
}

void Kmeans::loadBuffer(double *dataBuffer, int nRows, int nCols) {
	for (int i = 0; i < nRows; i++) {
		dataSet.push_back(Point((dataBuffer + i * nCols), nCols));
	}
}

void Kmeans::randomSetCentroids() {
	srand((unsigned int)time(NULL));
	map<int, bool> randomUsed;
	for (int i = 0; i < nClusters; i++) {
		int k = rand();
		while (randomUsed.find(k) != randomUsed.end()) k = rand();
		double *pointData = new double[nCols];
		for (int j = 0; j < dataSet[k].dimension; j++)
			pointData[j] = dataSet[k].data[j];
		centroids.push_back(Point(pointData, nCols));
		randomUsed[k] = true;
	}
	randomUsed.clear();
	map<int, bool>().swap(randomUsed);
}

double square(double a) { return a*a; }
void Kmeans::kmeans_clustering(int *label) {
	randomSetCentroids();
	vector<PPoint>* clusters = new vector<PPoint>[nRows];
	while (1) {
		//Calculates the cluster of each point
		for (int point_idx = 0; point_idx < nRows; point_idx++) {			
			double minDist = -1;
			for (int centroid_idx = 0; centroid_idx < nClusters; centroid_idx++) {
				double dist = 0;
				for (int i = 0; i < nCols; i++)
					dist += square(dataSet[point_idx].data[i] - centroids[centroid_idx].data[i]);
				if (minDist == -1 || dist < minDist) {
					minDist = dist;
					label[point_idx] = centroid_idx;
				}
			}
			clusters[label[point_idx]].push_back(&dataSet[point_idx]);
		}
		//Update the centroids of clusters
		bool hasCentroidChanged = false;
		for (int centroid_idx = 0; centroid_idx < nClusters; centroid_idx++) {
			double *newData = new double[nCols];
			fill(newData, newData + nCols, 0);
			int nPoints = clusters[centroid_idx].size();
			for (PPoint pPoint : clusters[centroid_idx]) {
				for (int i = 0; i < nCols; i++) 
					newData[i] += (pPoint->data[i])/nPoints;
			}
			for (int i = 0; i < nCols; i++) 
				if (newData[i] != centroids[centroid_idx].data[i]) {
					hasCentroidChanged = true;
					centroids[centroid_idx].data[i] = newData[i];
				}
			free(newData);			
		}
		if (!hasCentroidChanged) break;
	}	
	for (int i = 0; i < nRows; i++) {
		clusters[i].clear();
		vector<PPoint>().swap(clusters[i]);
	}
	free(clusters);
}

void Kmeans::clearData() {
	for (auto it = dataSet.begin(); it != dataSet.end(); it++) {
		PPoint pPoint = &(*it);
		if (pPoint != NULL) {
			delete (pPoint);
			pPoint = NULL;
		}
	}
	dataSet.clear(); //swap in the destructor
	for (auto it = centroids.begin(); it != centroids.end(); it++) {
		PPoint pPoint = &(*it);
		if (pPoint != NULL) {
			free(pPoint->data);
			delete (pPoint);
			pPoint = NULL;
		}	
	}
	centroids.clear();
}
