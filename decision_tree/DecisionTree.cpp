#include "DecisionTree.h" 

template<typename T>
DecisionTree<T>::DecisionTree() : decisionTree_(NULL) {

}

template<typename T>
DecisionTree<T>::~DecisionTree() {
	clear();
}

template<typename T>
void DecisionTree<T>::train(T *dataBuffer, int nPoints, int nAttributes, int *label, int *nChildren, int nClusters, int dtree_type) {
	clear();
	nPoints_ = nPoints;
	nAttributes_ = nAttributes;
	nClusters_ = nClusters;
	nChildren_ = nChildren;
	loadBuffer(dataBuffer, label);
	bool *attribute_used = new bool[nAttributes_];
	fill(attribute_used, attribute_used + nAttributes_, false);
	if (dtree_type == ID3) {
		decisionTree_ = buildTree_ID3(dataSet_, attribute_used);
	}
	else {
		decisionTree_ = buildTree_C45(dataSet_, attribute_used);
	}
}

template<typename T>
void DecisionTree<T>::loadBuffer(T *dataBuffer, int *label) {
	for (int i = 0; i < nPoints_; i++) 
		dataSet_.push_back(new Point(dataBuffer + i * nAttributes_, label[i]));	
}

//Build Tree by ID3
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
				child_count[(int)pPoint->pointData_[i]]++;
				child_sample_count[(int)pPoint->pointData_[i]][pPoint->label_]++;
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
			child_sample_count = new int*[2];
			for (int j = 0; j < 2; j++) {
				child_sample_count[j] = new int[nClusters_];				
			}
			double *elements = new double[pPoints.size()];
			for (int j = 0; j < pPoints.size(); j++) {
				elements[j] = pPoints[j]->pointData_[i];
			}
			sort(elements, elements + pPoints.size());
			for (int j = 0; j < pPoints.size() - 1; j++) { //Find the best place to cut the elements into two sets
				fill(child_count, child_count + 2, 0);
				for (int k = 0; k < 2; k++)
					fill(child_sample_count[k], child_sample_count[k] + nClusters_, 0);
				for (DecisionTree<T>::PPoint pPoint : pPoints) {
					pPoint->pointData_[i] <= elements[j] ? child_count[0]++ : child_count[1]++;
					pPoint->pointData_[i] <= elements[j] ? child_sample_count[0][pPoint->label_]++ : child_sample_count[1][pPoint->label_]++;
				}
				double entropy_cut = 0.0;
				for (int k = 0; k < 2; k++) {
					double entropy_child = 0.0;
					for (int l = 0; l < nClusters_; l++) {
						double p = (child_sample_count[k][l] * 1.0) / child_count[k];
						if (p != 0) entropy_child -= p * (log(p) / log(2));
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
				else if (i == 1 && pPoint->pointData_[best_attribute] > best_tag)
					child_pPoints.push_back(pPoint);
			}
			tree->children_[i] = buildTree_ID3(child_pPoints, attribute_used);
		}		
	}
	return tree;
}

//Build Tree by C4.5
template<typename T>
typename DecisionTree<T>::Tree DecisionTree<T>::buildTree_C45(vector<PPoint>& pPoints, bool * attribute_used) {
	if (pPoints.empty()) return NULL;
	bool sameLabel = true;
	for (DecisionTree<T>::PPoint pPoint : pPoints)
		if (pPoint->label_ != pPoints[0]->label_) {
			sameLabel = false;
			break;
		}
	if (sameLabel) { //All the points has the same label, return
		DecisionTree<T>::Tree tree = new DecisionTree<T>::TNode(pPoints, -1);
		tree->label_ = pPoints[0]->label_;
		return tree;
	}
	//Calculates the information entropy of the total dataset
	double info_entropy = 0.0;
	int *sample_count = new int[nClusters_]; //sample_count[i] means the number of points in "i"th cluster
	fill(sample_count, sample_count + nClusters_, 0);
	for (DecisionTree<T>::PPoint pPoint : pPoints) 
		sample_count[pPoint->label_]++;
	for (int i = 0; i < nClusters_; i++) {
		double p = (sample_count[i] * 1.0) / pPoints.size();
		if (p != 0) info_entropy -= p * (log(p) / log(2));
	}
	//C4.5 calculates the maximum information gain ratio
	int best_attribute = -1;
	double max_info_gain_ratio;
	double best_tag;
	//Calculates the information gain ratio of every attribute
	for (int i = 0; i < nAttributes_; i++) {
		if (attribute_used[i]) continue;
		int *category_count; //category_count[a] means the number of points in category a of the attribute
		int **category_sample_count; //category_sample_count[a][b] means the number of points in category a of the attribute and in cluster b
		if (nChildren_[i] != -1) { //The attribute has discrete value
			//Calculates the entropy of this attribute
			category_count = new int[nChildren_[i]];
			fill(category_count, category_count + nChildren_[i], 0);
			category_sample_count = new int *[nChildren_[i]];
			for (int j = 0; j < nChildren_[i]; j++) {
				category_sample_count[j] = new int[nClusters_];
				fill(category_sample_count[j], category_sample_count[j] + nClusters_, 0);
			}
			for (DecisionTree<T>::PPoint pPoint : pPoints) {
				category_count[(int)pPoint->pointData_[i]]++;
				category_sample_count[(int)pPoint->pointData_[i]][pPoint->label_]++;
			}
			double entropy = 0.0;
			for (int j = 0; j < nChildren_[i]; j++) {
				double category_entropy = 0.0;
				for (int k = 0; k < nClusters_; k++) {
					double p = (category_sample_count[j][k] * 1.0) / category_count[j];
					category_entropy -= p * (log(p) / log(2));
				}
				entropy += (category_entropy * category_count[j]) / pPoints.size();
			}
			//Calculates informaiton gain
			double info_gain = info_entropy - entropy;
			//Calculates information gain ratio
			double split_info = 0.0;
			for (int j = 0; j < nChildren_[i]; j++) {
				double p = (category_count[j] * 1.0) / pPoints.size();
				if (p != 0) split_info -= p * (log(p) / log(2));
			}
			double info_gain_ratio = info_gain / split_info;
			if (best_attribute == -1 || info_gain_ratio > max_info_gain_ratio) {
				best_attribute = i;
				max_info_gain_ratio = info_gain_ratio;
			}
		}
		else { //The attribute has continuous value
			category_count = new int[2];
			category_sample_count = new int *[2];
			for (int j = 0; j < 2; j++) {
				category_sample_count[j] = new int[nClusters_];
			}
			double *elements = new double[pPoints.size()];
			for (int j = 0; j < pPoints.size(); j++) {
				elements[j] = pPoints[j]->pointData_[i];
			}
			sort(elements, elements + pPoints.size());
			//Find the best place to cut the points into two category
			int best_cut = -1;
			double best_cut_info_gain_ratio;
			for (int j = 0; j < pPoints.size() - 1; j++) { //cut between j and (j + 1)				
				fill(category_count, category_count + 2, 0);
				for (int k = 0; k < 2; k++) {
					fill(category_sample_count[k], category_sample_count[k] + nClusters_, 0);
				}
				for (DecisionTree<T>::PPoint pPoint : pPoints) {
					if (pPoint->pointData_[i] <= elements[j]) {
						category_count[0]++;
						category_sample_count[0][pPoint->label_]++;
					}
					else {
						category_count[1]++;
						category_sample_count[1][pPoint->label_]++;
					}
				}
				//Calculates the entropy of the cut between j & (j + 1)
				double cut_entropy = 0.0;
				for (int k = 0; k < 2; k++) {
					double category_entropy = 0.0;
					for (int l = 0; l < nClusters_; l++) {
						double p = (category_sample_count[k][l] * 1.0) / category_count[k];
						if (p != 0) category_entropy -= p * (log(p) / log(2));
					}
					cut_entropy += (category_entropy * category_count[k]) / pPoints.size();
				}
				//Calculates the information gain of this cut
				double info_gain = info_entropy - cut_entropy;
				//Calculates the information gain ratio of this cut
				double split_info = 0.0;
				for (int k = 0; k < 2; k++) {
					double p = (category_count[k] * 1.0) / pPoints.size();
					if (p != 0) split_info -= p * (log(p) / log(2));
				}
				double cut_info_gain_ratio = info_gain / split_info;
				if (best_cut == -1 || cut_info_gain_ratio > best_cut_info_gain_ratio) {
					best_cut = j;
					best_cut_info_gain_ratio = cut_info_gain_ratio;
				}
			}
			if (best_attribute == -1 || best_cut_info_gain_ratio > max_info_gain_ratio) {
				best_attribute = i;
				max_info_gain_ratio = best_cut_info_gain_ratio;
				best_tag = elements[best_cut] / 2 + elements[best_cut + 1] / 2;
			}
		}
	}
	if (best_attribute == -1) { //All the attributes has been used, return
		int *sample_count = new int[nClusters_];
		fill(sample_count, sample_count + nClusters_, 0);
		//Find the label of cluster with maximum number of points in it
		int maxLabel;
		int maxLabelCount = 0;
		for (DecisionTree<T>::PPoint pPoint : pPoints) {
			if (++sample_count[pPoint->label_] > maxLabelCount) {
				maxLabel = pPoint->label_;
				maxLabelCount = sample_count[pPoint->label_];
			}
		}
		DecisionTree<T>::Tree tree = new DecisionTree<T>::TNode(pPoints, -1);
		tree->label_ = maxLabel;
		return tree;
	}
	//Build Tree
	DecisionTree<T>::Tree tree;
	if (nChildren_[best_attribute] != -1) {
		tree = new DecisionTree<T>::TNode(pPoints, best_attribute, nChildren_[best_attribute], true);
		attribute_used[best_attribute] = true;
		for (int i = 0; i < nChildren_[best_attribute]; i++) {
			vector<DecisionTree<T>::PPoint> child_pPoints;
			for (DecisionTree<T>::PPoint pPoint : pPoints) {
				if (pPoint->pointData_[best_attribute] == i)
					child_pPoints.push_back(pPoint);
			}
			tree->children_[i] = buildTree_C45(child_pPoints, attribute_used);
		}
		attribute_used[best_attribute] = false;
	}
	else {
		tree = new DecisionTree<T>::TNode(pPoints, best_attribute, 2, false);
		tree->tag_ = best_tag;
		for (int i = 0; i < 2; i++) {
			vector<DecisionTree<T>::PPoint> child_pPoints;
			for (DecisionTree<T>::PPoint pPoint : pPoints) {
				if (i == 0 && pPoint->pointData_[best_attribute] < best_tag) {
					child_pPoints.push_back(pPoint);
				}
				else if (i == 1 && pPoint->pointData_[best_attribute] > best_tag) {
					child_pPoints.push_back(pPoint);
				}
			}
			tree->children_[i] = buildTree_C45(child_pPoints, attribute_used);
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
	if (nChildren_[tree->attribute_] != -1) 
		return dfs_predict(predictData, tree->children_[(int)predictData[tree->attribute_]]);
	else {
		DecisionTree<T>::Tree subtree = predictData[tree->attribute_] < tree->tag_ ? tree->children_[0] : tree->children_[1];
		return dfs_predict(predictData, subtree);
	}
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
