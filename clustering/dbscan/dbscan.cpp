#include "Dbscan.h"

#define NOISE_LABEL -1

#ifdef SQUARE
#undef SQUARE
#endif

#define SQUARE(a) ((a) * (a))

Dbscan::Dbscan() {

}

Dbscan::~Dbscan() {
	vector<PPoint>().swap(dataSet);
	vector<PPoint>().swap(corePoints);
}

int Dbscan::DBSCAN(double dist, int minPts, double *dataBuffer, int nPoints, int dimension, int *label) {
	this->dist = dist;
	this->minPts = minPts;
	this->nPoints = nPoints;
	this->dimension = dimension;
	loadBuffer(dataBuffer);
	int nClusters = db_clustering(label);
	clear();
	return nClusters;
}

void Dbscan::loadBuffer(double *dataBuffer) {
	for (int i = 0; i < nPoints; i++) {
		dataSet.push_back(new Point(i, (dataBuffer + i * dimension)));
	}
}

int Dbscan::db_clustering(int *label) {
	findNeighborPoints();
	findCorePoints();
	return connectCorePoints(label);
}

void Dbscan::findNeighborPoints() {
	for (int i = 0; i < nPoints; i++) 
		for (int j = i + 1; j < nPoints; j++) {
			double distance = 0;
			for (int d = 0; d < dimension; d++) 
				distance += SQUARE(dataSet[i]->data[d] - dataSet[j]->data[d]);
			if (distance < SQUARE(dist)) {
				dataSet[i]->neighborPts.push_back(j);
				dataSet[j]->neighborPts.push_back(i);
			}
		}
}

void Dbscan::findCorePoints() {
	for (int i = 0; i < nPoints; i++) 
		if (dataSet[i]->neighborPts.size() >= minPts) {
			dataSet[i]->isCorePoint = true;
			corePoints.push_back(dataSet[i]);
		}
}

int Dbscan::connectCorePoints(int *label) {
	fill(label, label + nPoints, NOISE_LABEL);
	int cluster_count = 0;
	for (int i = 0; i < corePoints.size(); i++) {
		int corePointID = corePoints[i]->id;
		if (label[corePointID] != NOISE_LABEL) continue;
		//Use BFS to connect core points
		label[corePointID] = cluster_count;
		queue<PPoint> q;
		q.push(corePoints[i]);
		while (!q.empty()) {
			PPoint pPoint = q.front();
			q.pop();
			for (int neighborID : pPoint->neighborPts) {
				if (label[neighborID] == NOISE_LABEL) {
					label[neighborID] = cluster_count;
					if (dataSet[neighborID]->isCorePoint) q.push(dataSet[neighborID]);
				}
			}
		}	
		cluster_count++;
	}
	return cluster_count;
}

int Dbscan::isNoiseLabel(int label) {
	return (label == NOISE_LABEL) ? 1 : 0;
}

void Dbscan::clear() {
	for (auto it = dataSet.begin(); it != dataSet.end(); it++) {
		if (*it != NULL) {
			(*it)->neighborPts.clear();
			vector<int>().swap((*it)->neighborPts);
			delete (*it);
			*it = NULL;
		}
	}
	//swap in destructor
	dataSet.clear();
	corePoints.clear();
}
