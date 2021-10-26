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
        return "".join([text[indexs_[i]:indexs_[i + 1]].replace(old_word, new_word) if random.randint(0, 1) else text[indexs_[i]:indexs_[i + 1]]
                        for i in range(0, len(indexs_) - 1)])
    def find_word_2_replace(self,text):
        punctuation = "＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
        def isSymbol(inputString):
            return bool(re.match(r'[^\w]', inputString))

        def hasNumbers(inputString):
            return bool(re.search(r'\d', inputString))

        words_list = jieba.cut(text)
        return [w for w in words_list if isSymbol(w) == False and hasNumbers(w) == False and w not in punctuation and w in self.keyvectors.vocab]

    def get_similar_words(self,words,size=20,threshold=0.65):
        similar_words_dict={}
        for word in words:
            similar_words_cilin=self.cilin.get_similar_words(word)
            similar_words=[w for w,s in self.keyvectors.neighbours(word,size=size) if s>threshold and w in similar_words_cilin and w!=word]
            similar_words_dict[word]=similar_words
        return similar_words_dict
    def parse(self,text):
        words = self.find_word_2_replace(text)
        similar_words_dict=self.get_similar_words(words)
        similar_text=[]
        for word,similar_words in similar_words_dict.items():
            for sim_w in similar_words:
                new_text=self.text_word_replace(text,word,sim_w)
                if text!=new_text:
                    similar_text.append(new_text)
        return similar_text

if __name__ == '__main__':
    wordreplace=WordReplace()
    text = "12213它包含中文标点符号，不sdfdsf223333ddd23包括用作停止的标点符号。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
    similar_texts=wordreplace.parse(text)
    for t in similar_texts:
        print(t)
