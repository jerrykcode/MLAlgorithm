cmake_minimum_required(VERSION 2.8)
project(DTDebug)
add_executable(DTDebug dtree_debug.cpp
		       ../DecisionTree.cpp)
add_definitions(-DDEBUG)
include_directories(../)
