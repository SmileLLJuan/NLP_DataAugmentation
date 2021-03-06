# -*- coding: utf-8 -*-
'''
@author: yaleimeng@sina.com
@license: (C) Copyright 2017
@desc:    使用评测数据，计算词语相似度以及皮尔逊系数。本混合方法的皮尔逊系数0.854左右
@DateTime: Created on 2017/12/28, at 18:28 by PyCharm
'''

from codes.word_similarity.Hybrid_Sim import HybridSim
from codes.word_similarity.Pearson import *

# 30个  评测词对中的左侧词
MC30_A = ['轿车', '宝石', '旅游', '男孩子', '海岸', '庇护所', '魔术师', '中午', '火炉', '食物', '鸟', '鸟', '工具', '兄弟', '起重机', '小伙子',
        '旅行', '和尚', '墓地', '食物', '海岸', '森林', '岸边', '和尚', '海岸', '小伙子', '琴弦', '玻璃', '中午', '公鸡']
# 30个  评测词对中的右侧词
MC30_B = ['汽车', '宝物', '游历', '小伙子', '海滨', '精神病院', '巫师', '正午', '炉灶', '水果', '公鸡', '鹤', '器械', '和尚', '器械', '兄弟',
         '轿车', '圣贤', '林地', '公鸡', '丘陵', '墓地', '林地', '奴隶', '森林', '巫师', '微笑', '魔术师', '绳子', '航行']

# 人工评定的相似度列表。
MC30_C = [0.98, 0.96, 0.96, 0.94, 0.925, 0.9025, 0.875, 0.855, 0.7775, 0.77, 0.7625, 0.7425, 0.7375, 0.705, 0.42, 0.415,
         0.29, 0.275, 0.2375, 0.2225, 0.2175, 0.21, 0.1575, 0.1375, 0.105, 0.105, 0.0325, 0.0275, 0.02, 0.02]

RG35_A = ['水果', '署名', '汽车', '高地', '大笑', '庇护所', '庇护所', '墓地', '男孩子', '垫子', '庇护所', '大笑', '男孩', '汽车', '护堤',
            '海滨', '鸟', '火炉', '鹤', '山岗', '墓地', '玻璃', '魔术师', '圣人', '圣贤', '山岗', '绳索', '玻璃', '大笑', '农奴',
            '署名', '森林', '雄鸡', '靠枕', '墓地']

RG35_B = ['火炉', '海滨', '巫师', '火炉', '器械', '水果', '和尚', '精神病院', '公鸡', '宝物', '墓地', '小伙子', '圣人', '垫子', '海滨',
            '航行', '树林', '器械', '公鸡', '树林', '坟堆', '珠宝', '圣贤', '巫师', '圣人', '斜坡', '绳子', '杯子', '微笑', '奴隶',
            '签名', '树林', '公鸡', '枕头', '墓园']

RG35_C = [0.0125, 0.015, 0.0275, 0.035, 0.045, 0.0475, 0.0975, 0.105, 0.11, 0.1125, 0.1975, 0.22, 0.24, 0.2425, 0.2425, 0.305, 0.31,
         0.3425, 0.3525, 0.37,  0.4225, 0.445, 0.455, 0.615, 0.6525, 0.8225, 0.8525, 0.8625, 0.865, 0.865, 0.8975, 0.9125, 0.92, 0.96, 0.97]


if __name__ == '__main__':

    print('词林词汇量', len(HybridSim.ci_lin.vocab ),'\t知网词汇量', len(HybridSim.how_net.vocab))
    print('两者总词汇量',len(HybridSim.ci_lin.vocab | HybridSim.how_net.vocab),'\t重叠词汇量', len(HybridSim.Common))

    sim_list = []
    for w1, w2 in zip(MC30_A, MC30_B):
        hybrid = HybridSim.get_Final_sim(w1, w2)
        print('使用混合方法计算，相似度为：', hybrid)
        sim_list.append(hybrid)
    #print(sim_list)


    print('\n本方法皮尔逊系数为：',  cal_pearson(sim_list,MC30_C))  # 打印皮尔逊相关系数
