CC = g++
DEPS = DecisionTree.h DecisionTree.cpp
OBJ = DecisionTree.o DecisionTree_Test.o
TARGET = DTTest
CFLAGS = -DPRINT_TREE

DecisionTree.o : $(DEPS)
	$(CC) -c DecisionTree.cpp $(CFLAGS)

DecisionTree_Test.o : DecisionTree_Test.cpp $(DEPS)
	$(CC) -c DecisionTree_Test.cpp $(CFLAGS)

$(TARGET) : $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(CFLAGS)
