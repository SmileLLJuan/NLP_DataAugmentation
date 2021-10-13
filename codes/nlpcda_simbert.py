#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 11:09
# @Author  : lilijuan
# @File    : nlpcda_simbert.py
from nlpcda import Simbert
from time import time


def test_sing(simbert, N):
    """
    功能: 单元测试
    :param simbert:
    :return:
    """
    while True:
        text = input("\n输入: ")
        ss = time()
        synonyms = simbert.replace(sent=text, create_num=N)
        for line in synonyms:
            print(line)
        print("总耗时{0}ms".format(round(1000 * (time() - ss), 3)))


if __name__ == "__main__":
    config = {
        'model_path': '/mnt/disk1/lilijuan/corpus/trained_bert_models/chinese_roformer-sim-char_L-6_H-384_A-6',
        'device': 'cuda',
        'max_len': 32,
        'seed': 1
    }
    sim_bert = Simbert(config=config)
    test_sing(simbert=sim_bert, N=10)  # 单元测试