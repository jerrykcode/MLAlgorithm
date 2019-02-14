CC = g++
TARGET = DTDEBUG
OBJ = DecisionTree.o dtree_debug.o

DECISION_TREE_PATH = ..
INCLUDE = -I$(DECISION_TREE_PATH)
DEF = -DDEBUG
CFLAGS = $(INCLUDE) $(DEF)

DEPS = $(DECISION_TREE_PATH)/DecisionTree.h $(DECISION_TREE_PATH)/DecisionTree.cpp


$(TARGET) : $(OBJ)
	$(CC) -o $(TARGET) $(OBJ)

DecisionTree.o : $(DEPS)
	$(CC) -c $(DECISION_TREE_PATH)/DecisionTree.cpp $(CFLAGS)

dtree_debug.o : $(DEPS) dtree_debug.cpp
	$(CC) -c dtree_debug.cpp $(CFLAGS)
