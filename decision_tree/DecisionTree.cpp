#include "DecisionTree.h" 

template<typename T>
DecisionTree<T>::DecisionTree() : decisionTree_(NULL) {

}

template<typename T>
DecisionTree<T>::~DecisionTree() {
	clear();
}

template<typename T>
void DecisionTree<T>::train(T *dataBuffer, int nPoints, int nAttributes, int *label, int *nChildren, int nClusters) {
	clear();
	nPoints_ = nPoints;
	nAttributes_ = nAttributes;
	nClusters_ = nClusters;
	nChildren_ = nChildren;
	loadBuffer(dataBuffer, label);
	bool *attribute_used = new bool[nAttributes_];
	fill(attribute_used, attribute_used + nAttributes_, false);
	decisionTree_ = buildTree_ID3(dataSet_, attribute_used);
}

template<typename T>
void DecisionTree<T>::loadBuffer(T *dataBuffer, int *label) {
	for (int i = 0; i < nPoints_; i++) 
		dataSet_.push_back(new Point(dataBuffer + i * nAttributes_, label[i]));	
}

template<typename T>
typename DecisionTree<T>::Tree DecisionTree<T>::buildTree_ID3(vector<PPoint>& pPoints, bool *attribute_used) {
	if (pPoints.empty()) return NULL;
	bool sameLabel = true;
	for (DecisionTree<T>::PPoint pPoint : pPoints)
		if (pPoint->label_ != pPoints[0]->label_) {
			sameLabel = false;
			break;
		}
	//All the points has the same label, return
	if (sameLabel) {
		DecisionTree<T>::Tree tree = new DecisionTree<T>::TNode(pPoints, -1);
		tree->label_ = pPoints[0]->label_;
		return tree;
	}
	//ID3 calculate max information gain
	//The attribute with the minimum entropy has the max information gain
	int best_attribute = -1;
	double best_attribute_entropy = -1;
	double best_tag = 0.0;
	for (int i = 0; i < nAttributes_; i++) {
		if (attribute_used[i]) continue; 		
		int *child_count; //child_count[a] means the number of points with the attribute value a
		int **child_sample_count; //child_sample_count[a][b] means the number of points with the attribute value a and in  clusster b
		if (nChildren_[i] != -1) { //If the attribute has discrete value
			child_count = new int[nChildren_[i]];
			fill(child_count, child_count + nChildren_[i], 0);
			child_sample_count = new int*[nChildren_[i]];
			for (int j = 0; j < nChildren_[i]; j++) {
				child_sample_count[j] = new int[nClusters_];
				fill(child_sample_count[j], child_sample_count[j] + nClusters_, 0);
			}
			for (DecisionTree<T>::PPoint pPoint : pPoints) {
				child_count[pPoint->pointData_[i]]++;
				child_sample_count[pPoint->pointData_[i]][pPoint->label_]++;
			}
			double entropy = 0.0;
			for (int j = 0; j < nChildren_[i]; j++) {
				double entropy_child = 0.0;
				for (int k = 0; k < nClusters_; k++) {
					double p = (child_sample_count[j][k] * 1.0) / child_count[j];
					entropy_child -= p * (log(p) / log(2));
				}
				entropy += (child_count[j] * entropy_child) / pPoints.size();
			}
			free(child_count);
			for (int j = 0; j < nChildren_[i]; j++)
				free(child_sample_count[j]);
			free(child_sample_count);
			if (best_attribute == -1 || entropy < best_attribute_entropy) {
				best_attribute = i;
				best_attribute_entropy = entropy;
			}
		}
		else { //The attribute has continuous value
			int best_cut = -1;
			double best_cut_entropy = -1;
			child_count = new int[2];
			fill(child_count, child_count + 2, 0);
			child_sample_count = new int*[2];
			for (int j = 0; j < 2; j++) {
				child_sample_count[i] = new int[nClusters_];
				fill(child_sample_count[i], child_sample_count[i] + nClusters_, 0);
			}
			double *elements = new double[pPoints.size()];
			for (int j = 0; j < pPoints.size(); j++) {
				elements[j] = pPoints[j]->pointData_[i];
			}
			sort(elements, elements + pPoints.size());
			for (int j = 0; j < pPoints.size() - 1; j++) { //Find the best place to cut the elements into two sets
				for (DecisionTree<T>::PPoint pPoint : pPoints) {
					pPoint->pointData_[i] <= elements[j] ? child_count[0]++ : child_count[1]++;
					pPoint->pointData_[i] <= elements[j] ? child_sample_count[0][pPoint->label_]++ : child_sample_count[1][pPoint->label_]++;
				}
				double entropy_cut = 0.0;
				for (int k = 0; k < 2; k++) {
					double entropy_child = 0.0;
					for (int l = 0; l < nClusters_; l++) {
						double p = (child_sample_count[k][l] * 1.0) / child_count[k];
						entropy_child -= p * (log(p) / log(2));
					}
					entropy_cut += (child_count[k] * entropy_child) / pPoints.size();
				}
				if (best_cut_entropy == -1 || entropy_cut < best_cut_entropy) {
					best_cut = j;
					best_cut_entropy = entropy_cut;
				}
			}
			if (best_attribute == -1 || best_cut_entropy < best_attribute_entropy) {
				best_attribute = i;
				best_attribute_entropy = best_cut_entropy;
				best_tag = elements[best_cut] / 2 + elements[best_cut + 1] / 2;
			}
		}		
	}
	//All the attributes had been used, return
	if (best_attribute == -1) {
		DecisionTree<T>::Tree tree = new DecisionTree<T>::TNode(pPoints, -1);
		int *sample_count = new int[nClusters_];
		fill(sample_count, sample_count + nClusters_, 0);
		//Find the label with the maximum appearence in the points
		int maxLabelCount = 0;
		int maxLabel = -1;
		for (DecisionTree<T>::PPoint pPoint : pPoints)
			if (++sample_count[pPoint->label_] > maxLabelCount) {
				maxLabelCount = sample_count[pPoint->label_];
				maxLabel = pPoint->label_;
			}
		free(sample_count);
		tree->label_ = maxLabel;
		return tree;
	}
	//Build tree
	DecisionTree<T>::Tree tree;
	if (nChildren_[best_attribute] != -1) {
		attribute_used[best_attribute] = true;
		tree = new DecisionTree<T>::TNode(pPoints, best_attribute, nChildren_[best_attribute], true);
		for (int i = 0; i < nChildren_[best_attribute]; i++) {
			vector<DecisionTree<T>::PPoint> child_pPoints;
			for (DecisionTree<T>::PPoint pPoint : pPoints)
				if (pPoint->pointData_[best_attribute] == i)
					child_pPoints.push_back(pPoint);
			tree->children_[i] = buildTree_ID3(child_pPoints, attribute_used);
		}
		attribute_used[best_attribute] = false;		
	}
	else {
		tree = new DecisionTree<T>::TNode(pPoints, best_attribute, 2, false);
		tree->tag_ = best_tag;
		for (int i = 0; i < 2; i++) {
			vector<DecisionTree<T>::PPoint> child_pPoints;
			for (DecisionTree<T>::PPoint pPoint : pPoints) {
				if (i == 0 && pPoint->pointData_[best_attribute] < best_tag)
					child_pPoints.push_back(pPoint);
				else if (pPoint->pointData_[best_attribute] > best_tag)
					child_pPoints.push_back(pPoint);
			}
			tree->children_[i] = buildTree_ID3(child_pPoints, attribute_used);
		}		
	}
	return tree;
}

template<typename T>
int DecisionTree<T>::predict(T * predictData) {
	return dfs_predict(predictData, decisionTree_);
}

template<typename T>
int DecisionTree<T>::dfs_predict(T * predictData, Tree tree) {
	if (tree == NULL) {
		return -1;
	}
	if (tree->label_ != -1) {
		return tree->label_;
	}
	return dfs_predict(predictData, tree->children_[predictData[tree->attribute_]]);
}

#ifdef PRINT_TREE
template<typename T>
void DecisionTree<T>::print(vector<string>& attribute_name, vector<vector<string>>& attribute_type_name, vector<string>& cluster_name) {
	dfs_print(attribute_name, attribute_type_name, cluster_name, 0, decisionTree_);
}

template<typename T>
void DecisionTree<T>::dfs_print(vector<string>& attribute_name, vector<vector<string>>& attribute_type_name, vector<string>& cluster_name, int level, Tree tree) {
	if (tree == NULL) {
		print_with_space("No data", level * 3);
		return;
	}
	if (tree->label_ != -1) {
		print_with_space(cluster_name[tree->label_], level * 3);
		return;
	}
	int attribute = tree->attribute_;
	for (int i = 0; i < nChildren_[attribute]; i++) {
		print_with_space((attribute_name[attribute] + "---" + attribute_type_name[attribute][i] + "?"), level * 5);
		dfs_print(attribute_name, attribute_type_name, cluster_name, level + 1, tree->children_[i]);
	}
}

template<typename T>
void DecisionTree<T>::print_with_space(string str, int nSpace) {
	for (int i = 0; i < nSpace; i++)
		putchar(' ');
	cout << str << endl;
}

#endif

template<typename T>
void DecisionTree<T>::clear() {
	for (auto it = dataSet_.begin(); it != dataSet_.end(); it++)
		if (*it != NULL) {
			delete (*it);
			*it = NULL;
		}
	dataSet_.clear();
	vector<PPoint>().swap(dataSet_);
	deleteTree(decisionTree_);
}

template<typename T>
void DecisionTree<T>::deleteTree(Tree tree) {
	if (tree == NULL) return;
	if (tree->label_ != -1) {
		delete tree;
		tree = NULL;
		return;
	}
	for (int i = 0; i < nChildren_[tree->attribute_]; i++) {
		deleteTree(tree->children_[i]);
	}
	delete tree;
	tree = NULL;
}