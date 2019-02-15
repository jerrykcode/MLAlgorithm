#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#ifdef PRINT_TREE
#include <string>
#endif
using namespace std;

#define N_CONTINUOUS_CATEGORY 2

typedef enum {
	ID3,
	C45,
	CART,
} DTREE_TYPE;

template<typename T>
class DecisionTree {
public:
	DecisionTree();
	~DecisionTree();

	void train(T *dataBuffer, int nPoints /*nRows*/, int nAttributes /*nCols*/, int *label, int *nChildren, int nClusters, int dtree_type);
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

	//Define tree node(ID3 & C4.5)
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

	//Ddfine binary tree node(CART)
	typedef struct BTNode {
		vector<PPoint> pPoints_;
		int attribute_;
		bool isAttributeDiscrete_;
		double tag_;
		int label_;
		struct BTNode *left_;
		struct BTNode *right_;
		BTNode() {}
		BTNode(vector<PPoint>& pPoints, int attribute) : pPoints_(pPoints), attribute_(attribute), label_(-1), tag_(0.0) {}
		BTNode(vector<PPoint>& pPoints, int attribute, bool isAttributeDiscrete) : pPoints_(pPoints), attribute_(attribute), 
			isAttributeDiscrete_(isAttributeDiscrete), label_(-1), tag_(0.0) {}
	} *BTree;

	void loadBuffer(T *dataBuffer, int *label);
	Tree buildTree_ID3(vector<PPoint>& pPoints, bool *attribute_used);
	Tree buildTree_C45(vector<PPoint>& pPoints, bool *attribute_used);
	BTree buildTree_CART(vector<PPoint>& pPoints, vector<bool>* attribute_category_used);
	int dfs_predict(T *predictData, Tree tree);
	int dfs_predict(T *predictData, BTree bTree);
	void clear();
	void deleteTree(Tree tree);

#ifdef PRINT_TREE
	void dfs_print(vector<string>& attribute_name, vector<vector<string>>& attribute_type_name, vector<string>& cluster_name, int level, Tree tree);
	void dfs_print(vector<string>& attribute_name, vector<vector<string>>& attribute_type_name, vector<string>& cluster_name, int level, BTree bTree);
	void print_with_space(string str, int nSpace);
#endif

	DTREE_TYPE dtree_type_;
	int nPoints_;
	int nAttributes_;
	vector<PPoint> dataSet_;
	int *nChildren_;
	int nClusters_;

	Tree decisionTree_;
	BTree decisionBTree_;
};
