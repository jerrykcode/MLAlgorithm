CC = g++
DEPS = pca.h
OBJ = pca.o test_eigen.o
CFLAGS = -DTEST_EIGEN

test_eigen : $(OBJ)
	$(CC) -o test_eigen $(OBJ) $(CFLAGS)

pca.o : $(DEPS) pca.cpp
	$(CC) -c pca.cpp $(CFLAGS)

test_eigen.o : $(DEPS) test_eigen.cpp
	$(CC) -c test_eigen.cpp $(CFLAGS)

