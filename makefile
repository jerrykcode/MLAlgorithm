# makefile
# make MLAlgorithm

CC = g++

OBJ = Kmeans.o MLAlgorithm.o

# python path
PY_PATH = d:\20170608down\python3.7.2
PY_DEPS = $(PY_PATH)\include
PY_LIBS = $(PY_PATH)\libs
LIB = python37

# kmeans path
KMEANS_PATH = clustering/kmeans

# include
INCLUDE = -I$(KMEANS_PATH) -I$(PY_DEPS)

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

# make MLAlgorithm.o
MLAlgorithm.o : MLAlgorithm.cpp $(KMEANS)
	$(CC) -c MLAlgorithm.cpp $(CFLAGS)
