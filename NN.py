import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale,StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
labelEncoder1 = LabelEncoder()
labelEncoder2 = LabelEncoder()
oneHotEncoder = OneHotEncoder(categorical_features=[1])   # dummy variable
data = pd.read_csv('Churn_Modelling.csv')
X = data.values[:, 3:-1]
X[:, 1] = labelEncoder1.fit_transform(X[:, 1])
X[:, 2] = labelEncoder2.fit_transform(X[:, 2])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]
y = data.values[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train.shape)
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs=100)


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
scores = classifier.evaluate(X_test, y_test)
print(scores)
