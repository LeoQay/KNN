from time import time
import numpy as np
from sklearn.metrics import accuracy_score
import transforms
from load_mnist import load
from nearest_neighbors_joblib import KNNClassifier
from cross_validation import kfold


SPACE = {
    'open': [(1, 2, 3), (1, 2, 3)],
    'close': [(1, 2, 3), (1, 2, 3)],
    'rotate': [(-15, -10, -5, 0, 5, 10, 15)],
    'shift': [(-3, -2, -1, 0, 1, 2, 3), (-3, -2, -1, 0, 1, 2, 3)],
    'dilate': [(1, 2, 3), (1, 2, 3)],
    'erode': [(1, 2, 3), (1, 2, 3)],
    'blur': [(1, 3, 5), (1, 3, 5)]
}

log_file = open('log_greedy.txt', 'a')


def cross_calc(model, n_folds, X, y, whats, add_self=False):
    stat = np.array([
        calc(model, X[fold[0], :], X[fold[1], :], y[fold[0]], y[fold[1]], whats, add_self)
        for fold in kfold(X.shape[0], n_folds)
    ])
    print(whats, file=log_file, flush=True)
    print('cross', stat, stat.mean(), file=log_file, flush=True)
    return stat.mean()


def calc(model: KNNClassifier, X_train, X_test, y_train, y_test, whats, add_self=False):
    joined = []
    if add_self:
        model.fit(X_train, y_train)
        joined.append(model.find_kneighbors(X_test, True))
    for what in whats:
        X_tr = transforms.do_transforms(X_train.reshape(-1, 28, 28), what).reshape(-1, 784)
        model.fit(X_tr, y_train)
        joined.append(model.find_kneighbors(X_test, True))
    acc = accuracy_score(y_test, model.predict_join(joined))
    print(whats, '|', acc, f'self={add_self}', file=log_file, flush=True)
    return acc


def greedy_search(model, X, y, folds, sub_space: list, start=None):
    def get_cur(current_):
        res = []
        ind = 0
        for name, args in sub_space:
            res.append([name])
            for arg in args:
                res[-1].append(arg[current_[ind]])
                ind += 1
        return res

    def calc(current_):
        cur_ = get_cur(current_)
        print(cur_, end=': ', file=log_file, flush=True)
        start = time()
        acc = np.zeros(len(folds))
        X_tr = transforms.do_transforms(X.reshape(-1, 28, 28), cur_).reshape(-1, 784)
        for i, fold in enumerate(folds):
            X_test, y_train, y_test = X[fold[1]], y[fold[0]], y[fold[1]]
            joined = []
            model.fit(X[fold[0]], y_train)
            joined.append(model.find_kneighbors(X_test, True))
            model.fit(X_tr[fold[0]], y_train)
            joined.append(model.find_kneighbors(X_test, True))
            acc[i] = accuracy_score(y_test, model.predict_join(joined))
        end = time()
        mean = acc.mean()
        print(end - start, mean, file=log_file, flush=True)
        return mean

    dims = []
    for _, args in sub_space:
        for arg in args:
            dims.append(len(arg))
    stat = np.full(fill_value=-1, shape=tuple(dims), dtype=float)
    if start is None:
        current = np.array([dim / 2 for dim in dims], dtype=int)
    else:
        current = np.array(start)

    stat[tuple(current)] = calc(current)
    while True:
        print(get_cur(current), stat[tuple(current)], file=log_file, flush=True)
        nexts = []
        for i, ind in enumerate(current):
            next1 = current.copy()
            next1[i] -= 1
            if next1[i] >= 0 and stat[tuple(next1)] < 0:
                nexts.append(next1)
            next2 = current.copy()
            next2[i] += 1
            if next2[i] < stat.shape[i] and stat[tuple(next2)] < 0:
                nexts.append(next2)
        scores = np.array([calc(nex) for nex in nexts])
        for i, score in enumerate(scores):
            stat[tuple(nexts[i])] = score
        arg = scores.argmax()
        if scores[arg] < stat[tuple(current)]:
            return get_cur(current)
        else:
            current = nexts[arg]


X, y, X_train, X_test, y_train, y_test = load()

knn = KNNClassifier(k=4, strategy='my_own', metric='cosine', weights=True)

print(file=log_file, flush=True)

'''greedy_search(knn, X, y, kfold(X.shape[0], 3),
              [
                  ('rotate', [(-5, 5)]),
              ])
'''

cross_calc(knn, 3, X, y, [
    [('rotate', -5), ('blur', 3, 3)],
    [('rotate', 10), ('blur', 3, 3)],
    [('rotate', 5), ('blur', 3, 3)],
    [('open', 2, 2), ('blur', 3, 3)],
    [('close', 2, 2), ('blur', 3, 3)],
    [('blur', 3, 3)]])


# calc(knn, X_train, X_test, y_train, y_test,
# [[('rotate', 10)], [('open', 2, 2)], [('close', 2, 2)], [('blur', 3, 3)]], False)
