from matplotlib import pyplot as plt, cm
from pca.utils.dataloader import read, get_face
from sklearn.decomposition import PCA
from pca.config.path import data_path, source_image_path, source_data_path
from skimage import io
import numpy as np
# 特征脸
# 2D

# if True:
if False:
    data = read('ORL_32_32.mat')
    gnd = data['gnd']
    fea = data['fea']
    plt.imshow(fea[0].reshape((32,32)).T, cmap=cm.gray)
    plt.show()
    print(gnd[0])
    print(gnd[10])

# if True:
if False:
    get_face()

if True:
# if False:
    pca = PCA(3)
    img = io.imread(source_image_path)
    pca.fit(img)
    cov_matritx = pca.get_covariance()
    lambda_EA = np.linalg.eigvals(cov_matritx)
    # print(lambda_EA)
    # io.imshow(lambda_EA)
    # io.show()
