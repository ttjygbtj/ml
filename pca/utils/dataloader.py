import os
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.decomposition import PCA


def read(file):
    return scio.loadmat(os.path.join(os.path.dirname(__file__), '..', 'data', file))


d = read('ex7data1.mat')
print(d.keys())
X = d['X']
print(type(X))
print(X.shape)
# print(X)
plt.scatter(X[:, 0], X[:, 1])
# plt.show()
pca = PCA(1)
re_X = pca.fit_transform(X)

print(type(re_X))
print(re_X.shape)
plt.plot(re_X[:, 0], re_X[:, 1])
plt.show()
# print(type(read('ORL_32_32.mat')))
# print(os.path.join(os.path.dirname(__file__), '..', 'data', 'ORL_32*32.mat'))
