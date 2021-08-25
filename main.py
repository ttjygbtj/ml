import os
import random
import shutil

from skimage import io
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np

lst = random.sample(range(1, 1001), 5)
lst.sort()
files = list(map(lambda d: "%06d.jpg" % d, lst))


def compress(ratios, files, random_states=['random']):
    """
    压缩图片,返回每个压缩比下的平均mse,反应压缩性能
    :param ratios:
    :param files:
    :param random_state:
    :return:
    """
    mses=[]
    for ratio in ratios:
        for random_state in random_states:
            mse=0
            save_path = os.path.join('result', random_state, "%d"%ratio)
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path)
            for file in files:
                I = io.imread(os.path.join('data', file))
                io.imshow(I)
                kmeans = KMeans(n_clusters=ratio, random_state=random_state)
                kmeans.fit(I.reshape(-1, 3))
                O = np.array(kmeans.cluster_centers_[kmeans.labels_], dtype='uint8').reshape(I.shape)
                io.imsave(os.path.join(save_path, "%s_%s"%(ratio, file)), O)
                mse+=mean_squared_error(I.reshape(-1, 3), O.reshape(-1, 3))
                io.imshow(O)
                io.show()
            mses.append(mse)
    return mses


ret = compress([8, 16, 32, 64, 128], files, ['random', ''])
print(ret)