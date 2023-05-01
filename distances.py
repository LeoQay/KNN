import numpy as np


def euclidean_distance(X, Y):
    answer = np.linalg.norm(X, axis=-1)[:, None] ** 2 + \
        np.linalg.norm(Y, axis=-1)[None, :] ** 2 - (2.0 * X) @ Y.T
    answer[answer < 0] = 0
    return np.sqrt(answer)


def cosine_distance(X, Y):
    X_norm = np.linalg.norm(X, axis=-1)
    Y_norm = np.linalg.norm(Y, axis=-1)
    X_norm[X_norm == 0] = 1
    Y_norm[Y_norm == 0] = 1
    result = X @ Y.T
    result /= X_norm[:, None]
    result /= Y_norm[None, :]
    return 1.0 - result
