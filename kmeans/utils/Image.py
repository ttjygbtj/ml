import math
import os
import shutil

import numpy as np
from skimage import io

from skimage.metrics import mean_squared_error
from xlwt import Workbook
import xlrd


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


def compress_image(ratios, files, func):
    """
    压缩图片,得到每个压缩比下图片的拼图
    :param ratios:
    :param files:
    :param KMeanses:
    :param names:
    :return:
    """
    for KMeans, name, args in func:
        save_path = os.path.join('result', name, 'jigsaw')
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        for file in files:
            I_path = os.path.join('data', file)
            I = io.imread(I_path)
            Os = I.copy()
            for ratio in ratios:
                kmeans = KMeans(n_clusters=ratio, **args)
                kmeans.fit(I.reshape(-1, 3))
                O = np.array(kmeans.cluster_centers_[kmeans.labels_], dtype='uint8').reshape(I.shape)
                Os = np.concatenate((Os, O), axis=1)
            num = len(ratios) + 1
            n = int(math.sqrt(num))
            W = I.shape[1]  # 一张图的宽度
            if num == n * n:
                w = n
                h = n
                Os_ = Os[:, :w * W].copy()
                for i in range(h - 1):
                    Os_ = np.vstack((Os_, Os[:, (i + 1) * w * W:(i + 2) * w * W]))
            elif num == n * (n + 1):
                w = n + 1
                h = n
                Os_ = Os[:, :w * W].copy()
                for i in range(h - 1):
                    Os_ = np.vstack((Os_, Os[:, (i + 1) * w * W:(i + 2) * w * W]))
            elif num < n * (n + 1):
                w = n + 1
                h = num // w
                Os_ = Os[:, :w * W].copy()
                for i in range(h - 1):
                    Os_ = np.vstack((Os_, Os[:, (i + 1) * w * W:(i + 2) * w * W]))
                Os_ = np.vstack((Os_, np.hstack(
                    (Os[:, (h * w) * W:], np.zeros((Os.shape[0], (n * (n + 1) - num) * W, Os.shape[2]))))))
            else:
                w = n + 1
                h = n
                Os_ = Os[:, :w * W].copy()
                for i in range(h - 1):
                    Os_ = np.vstack((Os_, Os[:, (i + 1) * w * W:(i + 2) * w * W]))
                Os_ = np.vstack((Os_, np.hstack(
                    (Os[:, (h * w) * W:], np.zeros((Os.shape[0], ((n + 1) * (n + 1) - num) * W, Os.shape[2]))))))
            Os_path = os.path.join(save_path, file)
            io.imsave(Os_path, Os_)
            io.imshow(Os_)
            io.show()


def compress_data(ratios, files, func):
    """
    压缩图片,计算每个压缩比下的平均mse
    :param ratios:
    :param files:
    :param random_state:
    :return:
    """
    for KMeans, name, args in func:
        wb = Workbook(encoding='utf-8')
        ws_mse = wb.add_sheet('mse')
        ws_compression_ratio = wb.add_sheet('compression_ratio')
        ws_mse.write(0, 0, '图片')
        ws_compression_ratio.write(0, 0, '图片')
        ws_mse.write(0, len(ratios) + 1, '原图')
        ws_compression_ratio.write(0, len(ratios) + 1, '原图')
        save_path = os.path.join('result', name)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        for ratio in ratios:
            ws_mse.write(0, ratios.index(ratio) + 1, ratio)
            ws_compression_ratio.write(0, ratios.index(ratio) + 1, ratio)
            mse = 0
            compression_ratio = 0
            save_path = os.path.join('result', name, "%d" % ratio)
            os.makedirs(save_path)
            for file in files:
                I_path = os.path.join('data', file)
                I = io.imread(I_path)
                kmeans = KMeans(n_clusters=ratio, **args)
                kmeans.fit(I.reshape(-1, 3))
                O = np.array(kmeans.cluster_centers_[kmeans.labels_], dtype='uint8').reshape(I.shape)
                O_path = os.path.join(save_path, "%s_%s.png" % (ratio, file))
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
