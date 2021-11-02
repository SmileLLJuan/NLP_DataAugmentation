#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/30 16:19
# @Author  : lilijuan
# @File    : api_test.py
from flask import Flask,request
from codes.simbert.text_Synonyms import SynonymsGenerator
from codes.chinese_eda.chinese_EDA import eda_get_synonyms
from codes.word_similarity.word_replace import WordReplace
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level='INFO')
app = Flask(__name__)
model_path="/mnt/disk1/lilijuan/corpus/trained_bert_models/chinese_roformer-sim-char_L-12_H-768_A-12"
synonyms=SynonymsGenerator(model_path=model_path)
synonyms_result= synonyms.gen_text_synonyms(text='帮我关一下台灯', n=10, threhold=0.9)
wordreplace=WordReplace()

@app.route('/')
def index():
    synonyms_result = synonyms.gen_text_synonyms(text='帮我关一下台灯', n=10, threhold=0.9)
    return 'hello world'

@app.route('/hello',methods=['POST', 'GET'])
def hello():
    if request.method == "GET":
        return "GET hello"
    else:
        return "post hello"
@app.route('/synonyms/v1/simbert',methods=['POST', 'GET'])
def simbert_synonyms():
    if request.method == "GET":
        return __name__
    else:
        data = request.json
        texts=data.get("texts")
        n = data.get("n") if "n" in data else 100
        k = data.get("k") if "k" in data else 20
        threhold = data.get("threhold") if "threhold" in data else 0.8
        results = synonyms.gen_synonyms(texts, n=n, k=k, threhold=threhold)
        logger.warning(results)
        return results
@app.route('/synonyms/v1/wordsreplace',methods=['POST', 'GET'])
def wordsreplace_synonyms():
    if request.method == "GET":
        return __name__
    else:
        data = request.json
        texts=data.get("texts")
        n = data.get("n") if "n" in data else 100
        k = data.get("k") if "k" in data else 20
        threhold = data.get("threhold") if "threhold" in data else 0.8
        results = wordreplace.pares_texts(texts,sample=k)
        logger.warning(results)
        return results

@app.route('/synonyms/v1/chineseEDA',methods=['POST', 'GET'])
def chineseEDA_synonyms():
    if request.method == "GET":
        return __name__
    else:
        data = request.json
        texts=data.get("texts")
        n = data.get("n") if "n" in data else 100
        k = data.get("k") if "k" in data else 20
        threhold = data.get("threhold") if "threhold" in data else 0.8
        results = eda_get_synonyms(texts, num_aug=k)
        logger.warning(results)
        return results
if __name__ == '__main__':

    app.run(host='0.0.0.0',port=8934)
