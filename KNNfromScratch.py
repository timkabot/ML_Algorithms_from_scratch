from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

import numpy as np

dataset = {"k": [[1, 2],
                 [2, 3],
                 [3, 1],
                 [7, 7]],

           "r": [[6, 5],
                 [5, 6],
                 [8, 6]]
           }

def knn(dataset,predict, k=3):
    distances = []
    for group in dataset:
        for features in dataset[group]:
            dist = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([dist, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    prediction = Counter(votes).most_common(1)
    return prediction
predict = [5,7]
print(knn(dataset, predict))
