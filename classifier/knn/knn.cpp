#include "knn.h"

#ifdef SQUARE
#undef SQUARE
#endif

#define SQUARE(a) ((a) * (a))

Knn::Knn() : k_(0), nPoints_(0), dimension_(0), hasDataLoaded_(false), p_predictPoint_(NULL) {

}

Knn::~Knn() {
	clearData();
}

void Knn::train(double *dataBuffer, int nPoints, int dimension, int *label) {
	nPoints_ = nPoints;
	dimension_ = dimension;
	clearData();
	label_ = label;
	loadBuffer(dataBuffer);	
}

void Knn::loadBuffer(double *dataBuffer) {
	for (int i = 0; i < nPoints_; i++) 
		dataSet_.push_back(new Point(i, (dataBuffer + i * dimension_)));
	hasDataLoaded_ = true;	
}

void Knn::clearData() {
	for (auto it = dataSet_.begin(); it != dataSet_.end(); it++) {
		if (*it != NULL) {
			delete (*it);
			*it = NULL;
		}
	}
	dataSet_.clear();
	vector<PPoint>().swap(dataSet_);
}

int Knn::predict(int k, double *pointBuffer, int pointBuffer_size) {
	if (!hasDataLoaded_) {
		return KNN_NODATA_ERROR;
	}
	if (pointBuffer_size != dimension_) {
		return KNN_DIMENSION_ERROR;	
	}
	k_ = k;
	p_predictPoint_ = new Point(-1, pointBuffer);	
	heap_ = new MinHeap(nPoints_);
	calc_distance();	
	int predict_label = mostFrequencyLabel();
	delete p_predictPoint_;
	p_predictPoint_ = NULL;
	heap_->clearHeap();
	delete heap_;
	heap_ = NULL;
	return predict_label;
}

void Knn::calc_distance() {
	for (int i = 0; i < nPoints_; i++) {
		double distance = 0.0;
		for (int j = 0; j < dimension_; j++) {
			distance += SQUARE(p_predictPoint_->data[j] - dataSet_[i]->data[j]);
		}
		heap_->insert(new Neighbor(distance, dataSet_[i]));
	}
}

int Knn::mostFrequencyLabel() {
	map<int, int> labelAppearTime;
	int maxAppearTime = 0, tempAppearTime;
	int maxAppearTimeLabel = 0;
	for (int i = 0; i < k_; i++) {
		PNeighbor pNeighbor = new Neighbor(0, NULL);
		heap_->top(pNeighbor);
		if (pNeighbor->p_neighborPoint != NULL) {
			int label = label_[pNeighbor->p_neighborPoint->id];
			if ((tempAppearTime = ++labelAppearTime[label]) > maxAppearTime) {
				maxAppearTime = tempAppearTime;
				maxAppearTimeLabel = label;
			}
		}
		delete pNeighbor;
	}
	labelAppearTime.clear();
	map<int, int>().swap(labelAppearTime);
	return maxAppearTimeLabel;
}

//MinHeap
Knn::MinHeap::MinHeap(int capacity) : size_(0), capacity_(capacity), elements_(new PNeighbor[capacity]) {
	for (int i = 0; i < capacity_; i++)
		elements_[i] = NULL;	
}

Knn::MinHeap::~MinHeap() {

}

void Knn::MinHeap::clearHeap() {
	for (int i = 0; i < size_; i++) {
		if (elements_[i] != NULL) {
			delete elements_[i];
			elements_[i] = NULL;	
		}
	}
	free(elements_);
}

void Knn::MinHeap::insert(PNeighbor pNeighbor) {
	if (size_ >= capacity_) return;
	int index = size_++;
	for (; index != 0 && elements_[index/2]->distance > pNeighbor->distance; index /= 2)
		elements_[index] = elements_[index/2];
	elements_[index] = pNeighbor;
}

void Knn::MinHeap::top(PNeighbor pNeighbor) {
	if (size_ == 0) return;
	if (size_ == 1) {
		*pNeighbor = *(elements_[--size_]);//elements_[o], size_ = 0
		delete elements_[size_];
		return;
	}
	int parent_idx, child_idx;
	PNeighbor topElement = elements_[0];
	PNeighbor temp = elements_[--size_];
	for (parent_idx = 0; parent_idx * 2 + 1 < size_; parent_idx = child_idx) {
		child_idx = parent_idx * 2 + 1;
		if (child_idx + 1 < size_ && elements_[child_idx + 1]->distance < elements_[child_idx]->distance) //right child exist and right child is smaller
			child_idx++;
		if (temp->distance <= elements_[child_idx]->distance) break;
		elements_[parent_idx] = elements_[child_idx];
	}
	elements_[parent_idx] = temp;
	*pNeighbor = *topElement;
	delete topElement;
}
