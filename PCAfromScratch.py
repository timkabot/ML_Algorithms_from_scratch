import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets
from sklearn import preprocessing

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = preprocessing.scale(X)
X = X[:, :5]
print(X.shape)
cov_mat = np.cov(X.T)
e_vals, e_vecs = np.linalg.eig(cov_mat)
print(e_vals)
var_exp = [(i / sum(e_vals))*100 for i in sorted(e_vals, reverse=True)]
print(np.cumsum(var_exp))
e_vecs = e_vecs[:2]
new_data =e_vecs.dot(X.T)
print(new_data.T.shape)
