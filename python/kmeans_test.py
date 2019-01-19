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

nClusters = 3
MLA.kmeans(nClusters, points, label)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(nRows):
    if label[i] == 0:
        red_x.append(points[i][0])
        red_y.append(points[i][1])
    elif label[i] == 1:
        blue_x.append(points[i][0])
        blue_y.append(points[i][1])
    elif label[i] == 2:
        green_x.append(points[i][0])
        green_y.append(points[i][1])
plt.scatter(red_x, red_y, c = 'r', marker = '.')
plt.scatter(blue_x, blue_y, c = 'b', marker = 'x')
plt.scatter(green_x, green_y, c = 'g', marker = 'D')
plt.show()

