"""
@Time    : 13/4/2020 13:44
@Author  : Young lee
@File    : main.py
@Project : machine_learning_in_action
"""
import numpy as np
import matplotlib.pyplot as plt

from PART_I.chapter2 import kNN


# 2.1.1 准备：使用 Python 导入数据
# group, labels = kNN.createDataSet()
# 2.1.2 实施kNN算法
# label = kNN.classify([0, 0], group, labels, 3)

# 2.2.1 准备数据：从文本文件中解析数据
# 打开datingTestSet2.text，书中提及的datingTestSet.txt 标签列格式不对应
# dating_data_set, dating_labels = kNN.file2matrix('./data/datingTestSet2.txt')

# # 2.2.2 分析数据：使用 Matplotlib 创建散点图
# # 使matplotlib正常显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 初始化画布
# fig = plt.figure()
# # 设置画布格式， ”111“表示，1x1网格图中的第一个子图
# ax = fig.add_subplot(111)
# ax.scatter(dating_data_set[:, 1], dating_data_set[:, 2], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
# # 添加x轴，y轴说明
# plt.xlabel('玩视频游戏所耗时间百分比')
# plt.ylabel('每周所消费的冰淇淋公升数')
# # 展示画布
# plt.show()

# 2.2.3 准备数据：归一化数值
# norm_mat, range, min_values = kNN.auto_norm(dating_data_set)

# 2.2.4 测试算法：作为完整程序验证分类器
# kNN.dating_class_test()

# 2.2.5 使用算法：构建完整可用系统
# kNN.classifyPerson()

# 2.3 示例：手写识别系统
kNN.handwritingClassTest()
