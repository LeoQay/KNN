import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import distances


class KNNClassifier:
    def __init__(self, k,
                 strategy='my_own',
                 metric='euclidean',
                 weights=False,
                 test_block_size=1000):
        """
        [] k - число ближайших соседей;
        [] strategy - алгоритм поиска ближайших соседей:
         -- 'my_own' - собственная реализация,
         -- 'brute' -
                    sklearn.neighbors.NearestNeighbors(algorithm='brute'),
         -- 'kd_tree' -
                    sklearn.neighbors.NearestNeighbors(algorithm='kd_tree'),
         -- 'ball_tree' -
                    sklearn.neighbors.NearestNeighbors(algorithm='ball_tree');
        [] metric - название метрики, по которой считается расстояние
                    между объектами:
         -- 'euclidean' - евклидова метрика,
         -- 'cosine' - косинусная метрика;
        [] weights - bool переменная:
         -- True - нужно использовать взвешенный метод,
         -- False - не нужно;
        [] test_block_size - размер блока данных для тестовой выборки;
        """
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.epsilon = 0.00001
        self.test_block_size = test_block_size
        self.train_classes = None
        self.y_train = None
        if self.strategy == 'my_own':
            self.X_train = None
        else:
            self.model = NearestNeighbors(algorithm=self.strategy,
                                          metric=self.metric)

    def fit(self, X, y):
        """
        [] X - обучающая выборка объектов;
        [] y - ответы объектов из обучающей выборки;
        """
        self.y_train = y
        if self.strategy == 'my_own':
            self.X_train = X
        else:
            self.model.fit(X)
        self.train_classes = np.unique(y)
        return self

    def predict(self, X):
        """
        [] X - тестовая выборка объектов;
        """
        if self.weights:
            dists, inds = self.find_kneighbors(X, True)
            return self.predict_weighted(dists, inds)
        else:
            inds = self.find_kneighbors(X, False)
            return self.predict_simple(inds)

    def predict_join(self, joined):
        if self.weights:
            dists = np.hstack([dist for dist, _ in joined])
            inds_in_dists = dists.argsort(axis=1)
            inds = np.take_along_axis(np.hstack([ind for _, ind in joined]), inds_in_dists, axis=1)[:, :self.k]
            dists = np.take_along_axis(dists, inds_in_dists, axis=1)[:, :self.k]
            return self.predict_weighted(dists, inds)
        else:
            return self.predict_simple(np.take_along_axis(np.hstack([ind for _, ind in joined]),
                                       np.hstack([dist for dist, _ in joined]).argsort(axis=1),
                                       axis=1)[:, :self.k])

    def predict_weighted(self, dists, inds):
        classes = np.take_along_axis(self.y_train[None, :], inds, axis=1)
        classes_counts = []
        votes = 1.0 / (dists + self.epsilon)
        for class_id in self.train_classes:
            classes_counts.append((votes * (classes == class_id)).sum(axis=1))
        ind = np.vstack(classes_counts).argmax(axis=0)
        return np.take_along_axis(self.train_classes, ind, axis=0)

    def predict_simple(self, inds):
        classes = np.take_along_axis(self.y_train[None, :], inds, axis=1)
        classes_counts = []
        for class_id in self.train_classes:
            classes_counts.append((classes == class_id).sum(axis=1))
        ind = np.vstack(classes_counts).argmax(axis=0)
        return np.take_along_axis(self.train_classes, ind, axis=0)
    
    def find_kneighbors(self, X, return_distance):
        """
        [] X - тестовая выборка объектвов;
        [] return_distance - переменная типа bool;
        """
        if self.strategy == 'my_own':
            return self.find_kneighbors_my_own(X, return_distance)
        else:
            return self.model.kneighbors(X, n_neighbors=self.k,
                                         return_distance=return_distance)
    
    def find_kneighbors_my_own(self, X, return_distance):
        def my_job(X_part):
            return self.find_kneighbors_my_own_job(X_part, return_distance)
        
        result = Parallel(n_jobs=3)([
                delayed(my_job)
                (X[start:min(start + self.test_block_size, X.shape[0]), :])
                for start in range(0, X.shape[0], self.test_block_size)
            ])
        if return_distance:
            return np.vstack([t[0] for t in result]), np.vstack([t[1] for t in result])
        else:
            return np.vstack(result)
    
    def find_kneighbors_my_own_job(self, X, return_distance):
        if self.metric == 'euclidean':
            dist = distances.euclidean_distance(X, self.X_train)
        elif self.metric == 'cosine':
            dist = distances.cosine_distance(X, self.X_train)
        else:
            raise TypeError
        if self.k < dist.shape[1]:
            ind_in_block = dist.argpartition(self.k, axis=1)[:, :self.k]
            block_k = np.take_along_axis(dist, ind_in_block, axis=1)
        else:
            block_k = dist
            rang = np.arange(0, dist.shape[1])
            ind_in_block = np.vstack([rang for _ in range(X.shape[0])])
        ind_in_block_k = block_k.argsort(axis=1)
        ind = np.take_along_axis(ind_in_block, ind_in_block_k, axis=1)
        if return_distance:
            return np.take_along_axis(block_k, ind_in_block_k, axis=1), ind
        else:
            return ind
