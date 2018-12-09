import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(a):
    return a * (1 - a)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(a):
    return 1 - np.power(a, 2)


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(a):
    return (a > 0) * 1.0


activation_sigmoid = {
    'func': sigmoid,
    'derivative': sigmoid_derivative
}
# download the dataset

X, labels = sklearn.datasets.make_moons(200, noise=0.20)
y = labels.reshape(-1, 1)

def fit_net(X, layers, activation_funcs):
    theta = []
    bias = []

    # initialize parameters
    for i in range(len(layers) - 1):
        theta.append(np.random.random((layers[i], layers[i + 1])))
        bias.append(np.random.random((1, layers[i + 1])))

    # lr: learning rate for parameters updating
    lr = 0.1
    for j in range(20000):
        activations = []
        delta = []

        # FEED FORWARD PROPAGATION
        for i in range(len(layers) - 1):
            if i == 0:
                activations.append(activation_funcs[i]['func'](np.dot(X, theta[i]) + bias[i]))
            else:
                activations.append(activation_funcs[i]['func'](np.dot(activations[i - 1], theta[i]) + bias[i]))

        # BACKWARD PROPAGATION
        for i in range(len(layers) - 2, -1, -1):
            if i == len(layers) - 2:
                delta.append((activations[i] - y) * activation_funcs[i]['derivative'](activations[i]))
            else:
                delta.append(delta[-1].dot(theta[i + 1].T) * activation_funcs[i]['derivative'](activations[i]))

        delta = list(reversed(delta))

        # UPDATE PARAMETERS
        for i in range(len(layers) - 1):
            if i == 0:
                theta[i] = theta[i] - lr * X.T.dot(delta[i])
            else:
                theta[i] = theta[i] - lr * activations[i - 1].T.dot(delta[i])
            bias[i] = bias[i] - lr * delta[i].sum(axis=0, keepdims=True)

    return theta, bias


def predict(X, layers, theta, bias, activation_funcs):
    a = []
    for i in range(len(layers) - 1):
        if i == 0:
            a.append(activation_funcs[i]['func'](np.dot(X, theta[i]) + bias[i]))
        else:
            a.append(activation_funcs[i]['func'](np.dot(a[i - 1], theta[i]) + bias[i]))

    return a[len(layers) - 2]



# initial network
levels = [2, 5, 6,5,1]
activations = [activation_sigmoid] * 5
theta, bias = fit_net(X, levels, activations)
