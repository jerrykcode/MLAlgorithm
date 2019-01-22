import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import MLAlgorithm as MLA

def loadData(filePath):
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    data = []
    label = []
    maxLabel = 0
    for line in lines:
        lineData = line.strip().split(":")
        numbers = lineData[0].strip().split(",")
        data.append(numbers)
        label.append(int(lineData[1]))
        if int(lineData[1]) > maxLabel:
            maxLabel = int(lineData[1])
    return data, label, len(lines), maxLabel + 1

data, labelData, nRows, nClusters = loadData("labeled_data.txt")
points = np.asarray(data, dtype = "d")
label = np.asarray(labelData, dtype = "i")

MLA.knn_train(points, label)

k = 3
pointData = [1000, 999]
point = np.asarray(pointData, dtype = "d")
predict_label = MLA.knn_predict(k, point)

print(predict_label)

point_x = [[] for row in range(nClusters)]
point_y = [[] for row in range(nClusters)]
for i in range(nRows):
    point_x[label[i]].append(points[i][0])
    point_y[label[i]].append(points[i][1])

color = ['r', 'g', 'b', 'y', 'c', 'm']
for i in range(nClusters):
    plt.scatter(point_x[i], point_y[i], c = color[i], marker = '.')
    if i == predict_label:
        plt.scatter(1000, 999, c = color[i], marker = 'x')

plt.show()


