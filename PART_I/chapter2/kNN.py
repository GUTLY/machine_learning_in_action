"""
@Time    : 12/4/2020 13:55
@Author  : Young lee
@File    : kNN.py
@Project : machine_learning_in_action

k-Nearest Neighbor, kNN, k近邻算法
"""
import os
import operator

import numpy as np


def createDataSet():
    """生成数据集和标签"""
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    # 数据集，标签
    return group, labels


def classify(in_X, data_set, labels, k):
    """kNN算法实现

    :param in_X: 用于分类的输入向量 【1xN向量】
    :param data_set: 输入的训练样本集 【NxM矩阵】
    :param labels: 标签向量 【元素数目和矩阵dataSet的行数相同，1xM向量】
    :param k:最近邻居的数目 【奇数】
    :return: label : 输入in_X对应标签
    """
    # 数据集行数
    data_set_ize = data_set.shape[0]

    # 使用欧式距离
    # 将in_X沿y轴扩大data_set_ize倍，构成与data_set形状相同的矩阵，然后减去data_set得到diff_mat【差矩阵】
    diff_mat = np.tile(in_X, (data_set_ize, 1)) - data_set
    # 平方 差矩阵
    sq_diff_mat = diff_mat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    # 按行求和 得到 1xN 向量
    sq_distances = sq_diff_mat.sum(axis=1)
    # sq_distances 元素开方 得到 1xN 向量
    distances = sq_distances ** 0.5
    # 将distances中的元素从小到大排列，提取其对应的index(索引)，然后输出到sorted_dist_indicies
    sorted_dist_indicies = distances.argsort()

    class_count = {}
    # 选择距离最小的k个点【确定前k个距离最小元素所在的主要分类】
    for i in range(k):
        # 查找 sorted_dist_indicies 中第 i 的元素对应的标签
        vote_i_label = labels[sorted_dist_indicies[i]]
        # 字典的get()方法，返回指定键的值，如果值不在字典中返回0
        # 统计对应标签出现次数 { 标签：次数， 。。。}
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    # key = operator.itemgetter(1)根据字典的值进行排序
    # key = operator.itemgetter(0)根据字典的键进行排序
    # 根据标签次数 降序排序class_count
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别，即所要分类的类别
    return sorted_class_count[0][0]


def file2matrix(filename):
    """文件转矩阵

    :param filename: 文件名
    :return: 矩阵，标签向量
    """
    # 打开文件
    fr = open(filename)
    # 文件数据
    data = fr.readlines()
    # 关闭文件
    fr.close()
    # 文件行数
    number_of_lines = len(data)
    # 初始化矩阵
    mat = np.zeros((number_of_lines, 3))
    # 初始化标签
    label_vector = []
    index = 0
    for line in data:
        line = line.strip()
        list_from_line = line.split('\t')
        mat[index, :] = list_from_line[0:3]
        label_vector.append(int(list_from_line[-1]))
        index += 1
    return mat, label_vector


def auto_norm(data_set):
    """数据集归一化

    任意取值范围的特征值转化为0到1区间内的值：new_value = (old_value - min)/(max - min)
    其中min和max分别是数据集中的最小特征值和最大特征值
    :param data_set:待归一化矩阵
    :return:norm_data_set【归一化矩阵】，ranges【每一列(max - min)的向量】，min_vals【最小值向量】
    """
    # 按列取得最小、最大值
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    # 取得每列的取值范围：最大值-最小值
    ranges = max_vals - min_vals
    norm_data_set = np.zeros(np.shape(data_set))
    # 获取数据集列数
    m = data_set.shape[0]
    # 数据集每个值减去对应列的最小值
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    # 数据集除以对应列的取值范围(max - min)
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))

    return norm_data_set, ranges, min_vals


def dating_class_test():
    """约会数据分类测试"""
    # 测试数据比例
    ho_ratio = 0.10  # hold out 10%
    # 只选择样本数据集中前k个最相似的数据
    k = 3

    dating_data_mat, dating_labels = file2matrix('./data/datingTestSet2.txt')  # load data setfrom file
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    # 10%的测试数据的个数
    num_test_vecs = int(m * ho_ratio)
    # 分类错误个数
    errorCount = 0
    for i in range(num_test_vecs):
        # 前num_test_vecs个数据作为测试集，后m-num_test_vecs个数据作为训练集
        # k选择label数+1（结果比较好）
        classifier_result = classify(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], k)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i]))
        if (classifier_result != dating_labels[i]): errorCount += 1
    print("the total error rate is: %f" % (errorCount / float(num_test_vecs)))
    print(errorCount)


def classifyPerson():
    """约会网站预测函数"""

    # 预测结果列表
    result_list = ['不感兴趣', '有点喜欢', '非常喜欢']
    # 三维特征，用户输入
    percent_tats = float(input("玩视频游戏所消耗时间百分比："))
    fly_miles = float(input("每年飞行里程数："))
    ice_cream = float(input("每周消费的冰淇淋公升数："))
    # 数据集文件名
    filename = "./data/datingTestSet2.txt"
    # 打开并处理数据
    dating_data_mat, dating_labels = file2matrix(filename)
    # 训练集归一化
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # 输入转NumPy数组，用于预测
    in_arr = np.array([percent_tats, fly_miles, ice_cream])
    # 预测输入归一化
    normin_arr = (in_arr - min_vals) / ranges
    # 返回分类结果
    classifierResult = classify(normin_arr, norm_mat, dating_labels, 3)
    # 打印结果
    print("你可能对这个人 %s" % (result_list[classifierResult - 1]))


def img2vector(filename):
    """图像转向量"""
    # 向量初始化
    return_vect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def handwritingClassTest():
    # 标签集
    hw_labels = []
    # 训练数据列表
    training_file_list = os.listdir('./data/trainingDigits')
    # 训练数据集个数
    m = len(training_file_list)
    # 训练矩阵
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        # 文件名
        file_name_str = training_file_list[i]
        # 文件名形如‘2_19.txt’ 其中 2 为数据标签
        class_num = int(file_name_str.split('_')[0])
        hw_labels.append(class_num)
        # 训练集添加属性
        training_mat[i, :] = img2vector('./data/trainingDigits/%s' % file_name_str)
    # 测试数据列表
    test_file_list = os.listdir('./data/testDigits')
    # 预测错误个数
    error_count = 0
    # 测试数据集个数
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        # 测试数据真实标签
        class_num = int(file_name_str.split('_')[0])
        test_vector = img2vector('./data/trainingDigits/%s' % file_name_str)
        # 预测结果
        classifier_result = classify(test_vector, training_mat, hw_labels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_num))
        if (classifier_result != class_num): error_count += 1
    print("\nthe total number of errors is: %d" % error_count)
    print("\nthe total error rate is: %f" % (error_count / float(m_test)))
