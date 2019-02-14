from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append("..")
import MLAlgorithm as mla
import numpy as np

iris = load_iris()
# split data for train and test
data_train, data_test, label_train, label_test = train_test_split(iris['data'], iris['target'], random_state = 0)

nAttribute = iris.data.shape[1]
nChildren_data = []
for i in range(nAttribute):
    nChildren_data.append(-1)
nChildren = np.asarray(nChildren_data, dtype = "i")

for i in range(2):
    #train
    mla.decision_tree_train(data_train, label_train, nChildren, 3, i)

    #test
    right = 0
    wrong = 0
    for j in range(data_test.shape[0]):
        label = mla.decision_tree_predict(data_test[j])
        if label == label_test[j]:
            right = right + 1
        else:
            wrong = wrong + 1
    if i == 0:
        print("ID3 :")
    else:
        print("C4.5 :")
    print("right : ", end = "")
    print(right)
    print("wrong : ", end = "")
    print(wrong)
    print("rate : ", end = "")
    print(right / data_test.shape[0])

#train and predict by sklearn
clf = DecisionTreeClassifier()
clf = clf.fit(data_train, label_train)
right = 0
wrong = 0
for i in range(data_test.shape[0]):
    test = np.array([data_test[i]]) 
    label = clf.predict(test)
    if label == label_test[i]:
        right = right + 1
    else:
        wrong = wrong + 1
print("sklearn :")
print("right : ", end = "")
print(right)
print("wrong : ", end = "")
print(wrong)
print("rate : ", end = "")
print(right / data_test.shape[0])
