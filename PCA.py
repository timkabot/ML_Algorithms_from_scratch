import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets
from sklearn import preprocessing

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X)
pca = decomposition.PCA(n_components=3)
pca.fit(X)
print(pca.components_)
print(pca.explained_variance_)
X = pca.transform(X)

print(X)