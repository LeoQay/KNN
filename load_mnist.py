from time import time
import numpy as np


def load():
    start = time()
    X, y = np.load('mnist_784_X.npy', allow_pickle=True), np.load('mnist_784_y.npy', allow_pickle=True)
    X_train, X_test, y_train, y_test = X[:-10000, :], X[-10000:, :], y[:-10000], y[-10000:]
    end = time()

    print(end - start)
    print(type(X), X.shape)
    print(type(y), y.shape)
    print(X.dtype, y.dtype)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    return X, y, X_train, X_test, y_train, y_test
