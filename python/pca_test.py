import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import sys
sys.path.append("..")
import MLAlgorithm as MLA
import numpy as np

iris = load_iris()
data = iris.data

reduced_data = np.zeros((data.shape[0], 2))
MLA.pca_transform(2, data, reduced_data)

points_x, points_y = [], []
for i in range(data.shape[0]):
    points_x.append(reduced_data[i][0])
    points_y.append(reduced_data[i][1])

plt.scatter(points_x, points_y, marker = 'x')

skpca = PCA(n_components = 2)
skreduced_data = skpca.fit_transform(data)
skpoints_x, skpoints_y = [], []
for i in range(data.shape[0]):
    skpoints_x.append(skreduced_data[i][0])
    skpoints_y.append(skreduced_data[i][1])

plt.scatter(skpoints_x, skpoints_y, marker = '.')

plt.show()
