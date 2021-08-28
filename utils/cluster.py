import random
import numpy as np
from skimage.metrics import mean_squared_error


class lcKMeans():
    def __init__(self, n_clusters, init='r', max_iter=1):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter

    def fit(self, X):
        if self.init == 'r':
            idxes = random.sample(range(len(X)), self.n_clusters)
        elif self.init == 'l':
            idxes = np.linspace(0, len(X) - 1, self.n_clusters, dtype='int')
        U = X[idxes]
        for i in range(self.max_iter):
            C = [[u] for u in U]
            self.labels_ = []
            for x in X:
                md = mean_squared_error(x, U[0])
                mc = 0
                for i, u in enumerate(U):
                    d = mean_squared_error(x, u)
                    if md > d:
                        md = d
                        mc = i
                C[mc].append(x)
                self.labels_.append(mc)
            for i in range(self.n_clusters):
                U[i] = np.mean(np.array(C[i]))
        self.labels_ = np.array(self.labels_)
        self.cluster_centers_ = np.array(U)
