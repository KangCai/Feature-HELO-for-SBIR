# coding=utf-8

import numpy

a = numpy.array([[3, 0, 0], [0, 1, 0], [0, 1, 1]])
b = numpy.argwhere(a > 0)
print b
print '=' * 10

x_c = numpy.corrcoef(b.T)
# print x_c
eig_val, eig_vec = numpy.linalg.eig(x_c)
# print eig_val
print eig_vec
# print '=' * 10
# k = numpy.array([[0, 1], [1, 0]])
# print k.dot(eig_vec)
print '=' * 10


def pca(data_mat, top_n_feat=1):
    """
    pca特征维度压缩函数
    :param data_mat: 数据集矩阵
    :param top_n_feat: 需要保留的特征维度，即要压缩成的维度数
    :return:
    """
    # 求数据矩阵每一列的均值
    mean_val = numpy.mean(data_mat, axis=0)
    # 数据矩阵每一列特征减去该列的特征均值
    mean_removed = data_mat - mean_val
    # 计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
    cov_mat = numpy.cov(mean_removed, rowvar=False)
    # 计算协方差矩阵的特征值eig_val及对应的特征向量eig_vec
    eig_val, eig_vec = numpy.linalg.eig(numpy.mat(cov_mat))
    # argsort():对特征值矩阵进行由小到大排序，返回对应排序后的索引
    eig_val_ind = numpy.argsort(eig_val)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eig_val_ind = eig_val_ind[:-(top_n_feat + 1):-1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    red_eig_vec = eig_vec[:, eig_val_ind]
    print red_eig_vec
    a = numpy.arctan2(red_eig_vec[1], red_eig_vec[0])
    # 将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    low_dim_data_mat = mean_removed * red_eig_vec
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    recon_mat = (low_dim_data_mat * red_eig_vec.T) + mean_val
    # 返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return low_dim_data_mat, recon_mat

pca(b)