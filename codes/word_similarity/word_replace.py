#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/26 9:30
# @Author  : lilijuan
# @File    : word_replace.py
'''
中文词语替换
根据词典或词向量的方式替换文本中的词语'''
import os,re,random
from codes.synonyms.word2vec import KeyedVectors
from codes.word_similarity.cilin.V3.ciLin import CilinSimilarity
import codes.synonyms.jieba as jieba
class WordReplace(object):
    def __init__(self):
        curdir=os.path.dirname(__file__)
        _fin_wv_path = os.path.join(curdir, '../synonyms/data', 'words.vector.gz')
        _fin_stopwords_path = os.path.join(curdir, '../synonyms/data', 'stopwords.txt')
        kv = KeyedVectors()
        self.keyvectors = kv.load_word2vec_format(_fin_wv_path,binary=True,unicode_errors='ignore')
        # rr = self.keyvectors.neighbours("王国", 10)
        self.cilin=CilinSimilarity()
        # similar_words = self.cilin.get_similar_words("国王")

    def text_word_replace(self, text, old_word, new_word):  #文本词语替换
        indexs_ = [0] + [i.span()[0] for i in re.finditer(old_word, text)] + [len(text)]
        return text.replace(old_word,new_word)
        # return "".join([text[indexs_[i]:indexs_[i + 1]].replace(old_word, new_word) if random.randint(0, 1) else text[indexs_[i]:indexs_[i + 1]]
        #                 for i in range(0, len(indexs_) - 1)])
    def find_word_2_replace(self,text):
        punctuation = "＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
        def isSymbol(inputString):
            return bool(re.match(r'[^\w]', inputString))

        def hasNumbers(inputString):
            return bool(re.search(r'\d', inputString))

        words_list = jieba.cut(text)
        return [w for w in words_list if isSymbol(w) == False and hasNumbers(w) == False and w not in punctuation and w in self.keyvectors.vocab]

    def get_similar_words(self,words,size=25,threshold=0.5):
        similar_words_dict={}
        for word in words:
            similar_words_cilin=self.cilin.get_similar_words(word)
            similar_words_vector=self.keyvectors.neighbours(word,size=size)
            similar_words=[w for w,s in similar_words_vector if s>threshold and w in similar_words_cilin and w!=word]
            if len(similar_words)>0:
                similar_words_dict[word]=similar_words
        return similar_words_dict

    def parse(self,text,sample=10):
        words = self.find_word_2_replace(text)
        similar_words_dict=self.get_similar_words(words)
        similar_texts=[]
        # 一个句子中可以替换多个句子
        for i in range(sample):# 生成多少条
            new_text = text
            for word in random.sample(list(similar_words_dict), min(len(similar_words_dict),5)):# 每条可能随机替换5个词语
                new_word = similar_words_dict[word][random.randint(0, len(similar_words_dict[word]) - 1)]
                new_text = self.text_word_replace(new_text, word, new_word)
            if new_text != text and new_text not in similar_texts:
                similar_texts.append(new_text)
        return similar_texts
    def pares_texts(self,texts,sample=10):
        results=[]
        for text in texts:
            result_i=self.parse(text,sample)
            results.append({'text':text,'similar_texts':result_i})
        return {'code':200,'msg':'解析成功','result':results}

if __name__ == '__main__':
    import time
    wordreplace=WordReplace()
    texts = ['怎么开初婚未育证明', '帮我关一下台灯', '我想吃附近的火锅', '我们一起去打羽毛球吧', '把我的一个亿存银行安全吗', '微信和支付宝哪个好？', ]
    for text in texts:
        text = text.replace("\n", "")
        if len(text) > 2:
            similar_texts = wordreplace.parse(text, 20)
            print(text, similar_texts,"#"*20)
            for t in similar_texts:
                print(t)
