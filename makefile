# makefile
# make MLAlgorithm

CC = g++

OBJ = Kmeans.o Dbscan.o Knn.o Pca.o DecisionTree.o MLAlgorithm.o

# python path
PY_PATH = d:\20170608down\python3.7.2
PY_DEPS = $(PY_PATH)\include
PY_LIBS = $(PY_PATH)\libs
LIB = python37

# kmeans path
KMEANS_PATH = clustering/kmeans

# dbscan path
DBSCAN_PATH = clustering/dbscan

# knn path
KNN_PATH = classifier/knn

# pca path
PCA_PATH = dimensionality_reduction/pca

# decision tree path
DECISION_TREE_PATH = decision_tree

# include
INCLUDE = -I$(KMEANS_PATH) -I$(DBSCAN_PATH) -I$(KNN_PATH) -I$(PCA_PATH) -I$(DECISION_TREE_PATH) -I$(PY_DEPS)

# link
LINK = -L$(PY_LIBS) -l$(LIB)

# define
DEF = -D_hypot=hypot

# cflags
CFLAGS = $(DEF) $(INCLUDE) $(LINK)

# make target MLAlgorithm.pyd
MLAlgorithm : $(OBJ)
	$(CC) -o MLAlgorithm.pyd -shared $(OBJ) $(CFLAGS)

# make Kmeans.o
KMEANS = $(KMEANS_PATH)/Kmeans.h $(KMEANS_PATH)/Kmeans.cpp
Kmeans.o : $(KMEANS)
	$(CC) -c $(KMEANS_PATH)/Kmeans.cpp

# make Dbscan.o
DBSCAN = $(DBSCAN_PATH)/Dbscan.h $(DBSCAN_PATH)/Dbscan.cpp
Dbscan.o : $(DBSCAN)
	$(CC) -c $(DBSCAN_PATH)/Dbscan.cpp

# make Knn.o
KNN = $(KNN_PATH)/knn.h $(KNN_PATH)/knn.cpp
Knn.o : $(KNN)
	$(CC) -c $(KNN_PATH)/knn.cpp

# make Pca.o
PCA = $(PCA_PATH)/pca.h $(PCA_PATH)/pca.cpp
Pca.o : $(PCA)
	$(CC) -c $(PCA_PATH)/pca.cpp

# make decision tree
DECISION_TREE = $(DECISION_TREE_PATH)/DecisionTree.h $(DECISION_TREE_PATH)/DecisionTree.cpp
DecisionTree.o : $(DECISION_TREE)
	$(CC) -c $(DECISION_TREE_PATH)/DecisionTree.cpp

# make MLAlgorithm.o
MLAlgorithm.o : MLAlgorithm.cpp $(KMEANS) $(DBSCAN) $(KNN) $(PCA) $(DECISION_TREE)
	$(CC) -c MLAlgorithm.cpp $(CFLAGS)
