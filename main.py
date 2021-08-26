import math
import os
import random
import shutil
from skimage import io
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np
from xlwt import *

lst = random.sample(range(1, 1001), 1)
lst.sort()
files = list(map(lambda d: "%06d.jpg" % d, lst))


def size_format(size):
    if size < 1e3:
        return '%.1fB' % size
    elif size < 1e6:
        return '%.1fKB' % (size / 1e3)
    elif size < 1e9:
        return '%.1fMB' % (size / 1e6)
    elif size < 1e12:
        return '%.1fGB' % (size / 1e9)
    elif size < 1e15:
        return '%.1fTB' % (size / 1e12)


def mykmeans(ratios, files, KMeanses, names):
    pass


def compress_image(ratios, files, KMeanses, names):
    """
    压缩图片,得到每个压缩比下图片的拼图
    :param ratios:
    :param files:
    :param KMeanses:
    :param names:
    :return:
    """
    for KMeans, name in zip(KMeanses, names):
        save_path = os.path.join('result', name, 'jigsaw')
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        for file in files:
            I_path = os.path.join('data', file)
            I = io.imread(I_path)
            Os = I.copy()
            for ratio in ratios:
                kmeans = KMeans(n_clusters=ratio)
                kmeans.fit(I.reshape(-1, 3))
                O = np.array(kmeans.cluster_centers_[kmeans.labels_], dtype='uint8').reshape(I.shape)
                Os = np.concatenate((Os, O), axis=1)
            high = int(math.sqrt(len(ratios) + 1))
            unit = I.shape[1] * high
            Os_ = Os[:, :unit].copy()
            for i in range(high - 1):
                Os_ = np.vstack((Os_, Os[:, (i + 1) * unit:(i + 2) * unit]))
            Os_path = os.path.join(save_path, file)
            io.imsave(Os_path, Os_)
            io.imshow(Os_)
            io.show()


def compress_data(ratios, files, KMeanses, names):
    """
    压缩图片,计算每个压缩比下的平均mse
    :param ratios:
    :param files:
    :param random_state:
    :return:
    """
    for KMeans, name in zip(KMeanses, names):
        wb = Workbook(encoding='utf-8')
        ws_mse = wb.add_sheet('mse')
        ws_compression_ratio = wb.add_sheet('compression_ratio')
        ws_mse.write(0, 0, '图片')
        ws_compression_ratio.write(0, 0, '图片')
        ws_mse.write(0, len(ratios) + 1, '原图')
        ws_compression_ratio.write(0, len(ratios) + 1, '原图')
        for ratio in ratios:
            ws_mse.write(0, ratios.index(ratio) + 1, ratio)
            ws_compression_ratio.write(0, ratios.index(ratio) + 1, ratio)
            mse = 0
            compression_ratio = 0
            save_path = os.path.join('result', name, "%d" % ratio)
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path)
            for file in files:
                I_path = os.path.join('data', file)
                I = io.imread(I_path)
                kmeans = KMeans(n_clusters=ratio)
                kmeans.fit(I.reshape(-1, 3))
                O = np.array(kmeans.cluster_centers_[kmeans.labels_], dtype='uint8').reshape(I.shape)
                O_path = os.path.join(save_path, "%s_%s" % (ratio, file))
                io.imsave(O_path, O)
                mse_ = mean_squared_error(I.reshape(-1, 3), O.reshape(-1, 3))
                mse += mse_
                O_size = os.path.getsize(O_path)
                compression_ratio_ = O_size / os.path.getsize(I_path)
                compression_ratio += compression_ratio_
                ws_mse.write(files.index(file) + 1, ratios.index(ratio) + 1, mse_)
                ws_compression_ratio.write(files.index(file) + 1, ratios.index(ratio) + 1, size_format(O_size))

            mse, compression_ratio = np.array([mse, compression_ratio]) / len(files)
            ws_mse.write(len(files) + 1, ratios.index(ratio) + 1, mse)
            ws_compression_ratio.write(len(files) + 1, ratios.index(ratio) + 1, "%.2f%%" % (compression_ratio * 100))
        for file in files:
            ws_mse.write(files.index(file) + 1, 0, file)
            ws_compression_ratio.write(files.index(file) + 1, 0, file)
            I_path = os.path.join('data', file)
            I = io.imread(I_path)
            ws_mse.write(files.index(file) + 1, len(ratios) + 1, mean_squared_error(I.reshape(-1, 3), I.reshape(-1, 3)))
            ws_compression_ratio.write(files.index(file) + 1, len(ratios) + 1,
                                       size_format(os.path.getsize(I_path)))
        ws_mse.write(len(files) + 1, 0, 'avg')
        ws_compression_ratio.write(len(files) + 1, 0, 'avg')
        wb.save("%s.xls" % name)


def main():
    compress_data([2, 4, 8, 16, 32, 64, 128, 256], files, [KMeans], ['sklearn'])
    compress_image([2, 4, 8, 16, 32, 64, 128, 256], files, [KMeans], ['sklearn'])


if __name__ == '__main__':
    main()
