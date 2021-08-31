import random
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np
from utils.Image import compress_data, compress_image
from utils.cluster import lcKMeans




def main(labs):
    for lab in labs:
        if lab == 1:
            print('lab1: 聚类算法——k-means聚类')
            lst = random.sample(range(1, 1001), 1)
            lst.sort()
            files = list(map(lambda d: "%06d.jpg" % d, lst))
            ratios = np.linspace(8, 16, 3, dtype='int').tolist()
            # ratios = [1, 8, 16]
            func = [
                [lcKMeans, 'lcKmeans', {'init': 'r', 'max_iter': 10}]
                # [KMeans, 'sklearn', {}]
            ]
            compress_data(ratios, files, func)
            # compress_data([2, 4, 8, 16, 32, 64, 128, 256], files, [KMeans], ['sklearn'])
            compress_image(ratios, files, func)
        elif lab == 2:
            print('lab2: PCA 降维/特征提取')

if __name__ == '__main__':
    main([2])
    # np.linspace(1, 2, num=2, dtype='int')
