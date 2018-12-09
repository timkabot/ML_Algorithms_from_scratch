import numpy as np
import pandas as pd
from pprint import pprint
eps = np.finfo(float).eps
from numpy import log2 as log
dataset = {'Taste': ['Salty', 'Spicy', 'Spicy', 'Spicy', 'Spicy', 'Sweet', 'Salty', 'Sweet', 'Spicy', 'Salty'],
       'Temperature': ['Hot', 'Hot', 'Hot', 'Cold', 'Hot', 'Cold', 'Cold', 'Hot', 'Cold', 'Hot'],
       'Texture': ['Soft', 'Soft', 'Hard', 'Hard',' Hard', 'Soft', 'Soft', 'Soft', 'Soft', 'Hard'],
       'Eat':['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes']}
df = pd.DataFrame(dataset,columns=['Taste','Temperature','Texture','Eat'])

def find_entropy(df):
    Class = df.keys()[-1]
    entropy = 0  # Initialize the entropy
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction * log(fraction)
    return entropy

def find_gini(df):
    Class = df.keys()[-1]
    gini = 1
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        gini -= fraction**2
    return gini


def find_gini_attribute(df, attribute):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()
    variables = df[attribute].unique()
    gini = 0
    for variable in variables:
        score = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            score += fraction * fraction
        gini += (1.0 - score) * den/len(df)
    return abs(gini)

def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()
    variables = df[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        fraction2 = den / len(df)
        entropy2 += -fraction2 * entropy
    return abs(entropy2)

def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_gini(df)-find_gini_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]

def get_subtable(df, node,value):
    return df[df[node] == value].reset_index(drop=True)


def predict(inst, tree):

    for nodes in tree.keys():

        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break

    return prediction

def buildTree(df, tree=None):
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name

    # Here we build our decision tree

    # Get attribute with maximum information gain
    node = find_winner(df)
    print('Winner node', node)
    # Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])

    # Create an empty dictionary to create tree
    if tree is None:
        tree = {}
        tree[node] = {}

    # We make loop to construct a tree by calling this function recursively.
    # In this we check if the subset is pure and stops if it is pure.

    for value in attValue:

        subtable = get_subtable(df, node, value)
        clValue, counts = np.unique(subtable['Eat'], return_counts=True)
        if len(counts) == 1:  # Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable)  # Calling the function recursively

    return tree

tree = buildTree(df)
data = {'Taste':'Salty', 'Temperature':'Cold', 'Texture':'Hard'}
predict(data,tree)
