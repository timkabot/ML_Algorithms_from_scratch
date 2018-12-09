import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
data = pd.read_csv('titanic.csv')
data = data.drop(columns=['name','embarked','sex'],axis=1)
data = data.fillna(0)


X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train = preprocessing.scale(x_train)
X_test = preprocessing.scale(x_test)
model = SVC(C=30, gamma=0.01, kernel='rbf')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))