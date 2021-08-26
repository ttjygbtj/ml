import os
import random
import shutil
from skimage import io
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np
from xlwt import *

lst = random.sample(range(1, 1001), 5)
lst.sort()
files = list(map(lambda d: "%06d.jpg" % d, lst))


def mykmeans():
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
        Os = np.array([])
        save_path = os.path.join('result', name)
        for file, ratio in zip(files, ratios):
            I_path = os.path.join('data', file)
            I = io.imread(I_path)
            # io.imshow(I)
            kmeans = KMeans(n_clusters=ratio)
            kmeans.fit(I.reshape(-1, 3))
            O = np.array(kmeans.cluster_centers_[kmeans.labels_], dtype='uint8').reshape(I.shape)
            np.concatenate((O, Os))
            Os_path = os.path.join(save_path, "jigsaw%s" % (file))
            if os.path.exists(Os_path):
                shutil.rmtree(Os_path)
            os.makedirs(Os_path)
            io.imsave(Os_path, Os)
            io.imshow(Os)
            io.show()


def compress_data(ratios, files, KMeanses, names):
    """
    压缩图片,计算每个压缩比下的平均mse
    :param ratios:
    :param files:
    :param random_state:
    :return:
    """
    mses = []
    compression_ratios = []
    for KMeans, name in zip(KMeanses, names):
        wb = Workbook(encoding='utf-8')
        ws_mse = wb.add_sheet('mse')
        ws_compression_ratio = wb.add_sheet('compression_ratio')
        ws_mse.write(0, 0, '图片')
        ws_compression_ratio.write(0, 0, '图片')
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
                compression_ratio_ = os.path.getsize(O_path) / os.path.getsize(I_path)
                compression_ratio += compression_ratio_
                ws_mse.write(files.index(file) + 1, ratios.index(ratio) + 1, mse_)
                ws_compression_ratio.write(files.index(file) + 1, ratios.index(ratio) + 1, compression_ratio_)

            mse, compression_ratio = np.array([mse, compression_ratio]) / len(files)

            ws_mse.write(len(files) + 1, ratios.index(ratio) + 1, mse)
            ws_compression_ratio.write(len(files) + 1, ratios.index(ratio) + 1, compression_ratio)
            mses.append(mse)
            compression_ratios.append(compression_ratio)
        for file in files:
            ws_mse.write(files.index(file) + 1, 0, file)
            ws_compression_ratio.write(files.index(file) + 1, 0, file)
        ws_mse.write(len(files) + 1, 0, 'avg')
        ws_compression_ratio.write(len(files) + 1, 0, 'avg')
        wb.save("%s.xls" % name)
    return mses, compression_ratios


def main():
    ret1 = compress_data([8, 16], files, [KMeans], ['sklearn'])
    compress_image([8, 16], files, [KMeans], ['sklearn'])
    print(ret1)


if __name__ == '__main__':
    main()
