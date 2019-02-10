cmake_minimum_required(VERSION 2.8)
project(DTTest)
add_executable(DTTest DecisionTree.cpp
	       	      DecisionTree_Test.cpp
	       )
add_definitions(-D PRINT_TREE)
