import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib import cm
from sklearn.decomposition import PCA
from skimage import io

from pca.config.path import data_path, source_image_path, source_data_path


def read(file):
    return scio.loadmat(file)


def get_face(data=read(source_data_path)):
    fea = data['fea']
    O = []
    for i in range(10):
        o = fea[i * 10].reshape((32, 32)).T
        for j in range(1, 10):
            o = np.hstack((o, fea[i * 10 + j].reshape((32, 32)).T))
        O.append(o)
    os.makedirs(data_path, exist_ok=True)
    O = np.array(O)
    O.resize(320, 320)
    io.imsave(source_image_path, O)



