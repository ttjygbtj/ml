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
    pca = PCA()
    img = io.imread(source_image_path)
    pca.fit(img)
    engine = pca.components_
    io.imshow(img, cmap='bone')
    io.show()
    print(engine.shape)
    io.imshow(engine, cmap='bone')
    io.show()
