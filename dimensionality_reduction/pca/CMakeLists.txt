cmake_minimum_required(VERSION 2.8)
project(eigen)
add_executable(eigen 
	pca.cpp
    	test_eigen.cpp
)

add_definitions(-D TEST_EIGEN)

