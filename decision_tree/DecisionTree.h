#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#ifdef PRINT_TREE
#include <string>
#endif
using namespace std;

template<typename T>
class DecisionTree {
public:
	DecisionTree();
	~DecisionTree();

	void train(T *dataBuffer, int nPoints /*nRows*/, int nAttributes /*nCols*/, int *label, int *nChildren, int nClusters);
	int predict(T *predictData);

#ifdef PRINT_TREE
	void print(vector<string>& attribute_name, vector<vector<string>>& attribute_type_name, vector<string>& cluster_name);
#endif

private:
	typedef struct Point {
		T *pointData_;
		int label_;
		Point(T *pointData, int label) : pointData_(pointData), label_(label) {}
	} *PPoint;

	typedef struct TNode {
		vector<PPoint> pPoints_;
		int attribute_;
		bool isAttributeDiscrete_;
		double tag_;
		int label_;
		struct TNode ** children_;
		TNode() {}
		TNode(vector<PPoint>& pPoints, int attribute) : pPoints_(pPoints), attribute_(attribute), label_(-1), tag_(0.0) {}
		TNode(vector<PPoint>& pPoints, int attribute, int nChildren, bool isAttributeDiscrete) : pPoints_(pPoints), 
			attribute_(attribute), label_(-1), isAttributeDiscrete_(isAttributeDiscrete), tag_(0.0) {
			children_ = new struct TNode *[nChildren];
			for (int i = 0; i < nChildren; i++)
				children_[i] = NULL;
		}
	} *Tree;

	void loadBuffer(T *dataBuffer, int *label);
	Tree buildTree_ID3(vector<PPoint>& pPoints,  bool *attribute_used);
	int dfs_predict(T *predictData, Tree tree);
	void clear();
	void deleteTree(Tree tree);

#ifdef PRINT_TREE
	void dfs_print(vector<string>& attribute_name, vector<vector<string>>& attribute_type_name, vector<string>& cluster_name, int level, Tree tree);
	void print_with_space(string str, int nSpace);
#endif

	int nPoints_;
	int nAttributes_;
	vector<PPoint> dataSet_;
	int *nChildren_;
	int nClusters_;

	Tree decisionTree_;
};
