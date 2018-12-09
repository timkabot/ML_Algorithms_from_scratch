from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from pprint import pprint
import numpy as np
d = datasets.load_iris()
d2 = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.4)

baggingC = BaggingClassifier(KNeighborsClassifier(), max_samples=0.1, max_features=0.5)

baggingC.fit(X_train, y_train)
y_pred = baggingC.predict(X_test)
print('Bagging Classifier accuracy', accuracy_score(y_pred, y_test))

X_train, X_test, y_train, y_test = train_test_split(d2.data, d2.target, test_size=0.4)

baggingR = BaggingRegressor(LinearRegression(), n_estimators=20, max_samples=0.5, max_features=0.5)
baggingDecisionTree = BaggingRegressor(DecisionTreeRegressor(),max_samples=0.5,max_features=0.5, n_estimators=20)
baggingR.fit(X_train, y_train)
y_pred = baggingR.predict(X_test)
print('RMSE linear regression', np.sqrt(mean_squared_error(y_test,y_pred)))
baggingDecisionTree.fit(X_train, y_train)
y_pred = baggingDecisionTree.predict(X_test)
print('RMSE decision trees', np.sqrt(mean_squared_error(y_test,y_pred)))
