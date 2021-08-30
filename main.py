import random
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np
from utils.Image import compress_data, compress_image
from utils.cluster import lcKMeans

lst = random.sample(range(1, 1001), 1)
lst.sort()
files = list(map(lambda d: "%06d.jpg" % d, lst))


def main():
    ratios = np.linspace(8, 16, 3, dtype='int').tolist()
    # ratios = [1, 8, 16]
    func = [
        [lcKMeans, 'lcKmeans', {'init': 'r', 'max_iter': 10}]
        # [KMeans, 'sklearn', {}]
    ]
    compress_data(ratios, files, func)
    # compress_data([2, 4, 8, 16, 32, 64, 128, 256], files, [KMeans], ['sklearn'])
    compress_image(ratios, files, func)


if __name__ == '__main__':
    main()
    # np.linspace(1, 2, num=2, dtype='int')
