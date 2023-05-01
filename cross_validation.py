import numpy as np
from nearest_neighbors_joblib import KNNClassifier


def kfold(n, n_folds):
    indexes = np.arange(0, n)
    sizes = np.full(n_folds, n // n_folds)
    sizes[:n % n_folds] += 1
    coords = np.cumsum(np.array([sizes, sizes]), axis=1)
    coords[0, :] -= sizes
    mask = np.full(n, True)
    answer = []
    for coord in coords.T:
        coord = slice(coord[0], coord[1])
        mask[coord] = False
        answer.append((indexes[mask], indexes[coord]))
        mask[coord] = True
    return answer


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    EPSILON = 0.00001
    if cv is None:
        s = X.shape[0]
        cv = kfold(s, s if s < 3 else 3)
    stat = []
    for train, test in cv:
        X_train, X_test, y_train, y_test =\
            X[train, :], X[test, :], y[train], y[test]
        knn = KNNClassifier(k=k_list[-1], **kwargs)
        knn.fit(X_train, y_train)
        k_list.insert(0, 0)
        classes_id = np.unique(y_train)
        cv_result = []
        if kwargs['weights']:
            dists, indexes = knn.find_kneighbors(X_test, return_distance=True)
            votes = 1.0 / (dists + EPSILON)
            classes = np.take_along_axis(y_train[None, :], indexes, axis=1)
            classes_counts = np.zeros((classes.shape[0], classes_id.shape[0]), dtype=float)
            for i in range(1, len(k_list)):
                current_neighs = classes[:, k_list[i - 1]:k_list[i]]
                votes_neighs = votes[:, k_list[i - 1]:k_list[i]]
                for j, class_id in enumerate(classes_id):
                    classes_counts[:, j] += \
                        (votes_neighs * (current_neighs == class_id)).sum(axis=1)
                ind = classes_counts.argmax(axis=1)
                pred = np.take_along_axis(classes_id, ind, axis=0)
                cv_result.append((pred == y_test).sum() / y_test.shape[0])
        else:
            indexes = knn.find_kneighbors(X_test, return_distance=False)
            classes = np.take_along_axis(y_train[None, :], indexes, axis=1)
            classes_counts = np.zeros((classes.shape[0], classes_id.shape[0]), dtype=int)
            for i in range(1, len(k_list)):
                current_neighs = classes[:, k_list[i - 1]:k_list[i]]
                for j, class_id in enumerate(classes_id):
                    classes_counts[:, j] += (current_neighs == class_id).sum(axis=1)
                ind = classes_counts.argmax(axis=1)
                pred = np.take_along_axis(classes_id, ind, axis=0)
                cv_result.append((pred == y_test).sum() / y_test.shape[0])
        k_list.pop(0)
        stat.append(cv_result)
    stat = np.array(stat)
    return {k: stat[:, i].ravel() for i, k in enumerate(k_list)}