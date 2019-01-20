import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import MLAlgorithm as MLA

def loadData(filePath):
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    data = []
    for line in lines:
        numbers = line.strip().split(",")
        data.append(numbers)
    return data, len(lines)

data, nRows = loadData("data.txt")
points = np.asarray(data, dtype = "d")
label = (np.zeros(nRows)).astype(np.int32)

dist = 1000
minPts = 10
nClusters = MLA.dbscan(dist, minPts, points, label)

if nClusters == 0:
	print("no cluster!")	
else:
	point_x = [[] for row in range(nClusters)]
	point_y = [[] for row in range(nClusters)]

	noise_x = []
	noise_y = []

	for i in range(nRows):
		if (MLA.dbscan_isNoiseLabel(label[i]) == True):
			noise_x.append(points[i][0])
			noise_y.append(points[i][1])
		else:
			point_x[label[i]].append(points[i][0])
			point_y[label[i]].append(points[i][1])

	for i in range(nClusters):
		plt.scatter(point_x[i], point_y[i], marker = '.')
	plt.scatter(noise_x, noise_y, marker = 'x')
  	
	plt.show()

