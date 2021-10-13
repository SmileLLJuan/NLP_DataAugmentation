#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/30 16:19
# @Author  : lilijuan
# @File    : api_test.py
from flask import Flask,request
from codes.Text_Synonyms import SynonymsGenerator
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level='INFO')
app = Flask(__name__)
model_path="/mnt/disk1/lilijuan/corpus/trained_bert_models/chinese_roformer-sim-char_L-12_H-768_A-12"
synonyms=SynonymsGenerator(model_path=model_path)
synonyms_result= synonyms.gen_text_synonyms(text='帮我关一下台灯', n=10, threhold=0.9)
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
@app.route('/synonyms',methods=['POST', 'GET'])
def get_synonyms():
    if request.method == "GET":
        return "GET hello"
    else:
        data = request.json
        texts=data.get("texts")
        n = data.get("n") if "n" in data else 100
        k = data.get("k") if "k" in data else 20
        threhold = data.get("threhold") if "threhold" in data else 0.8
        results = synonyms.gen_synonyms(texts, n=n, k=k, threhold=threhold)
        logger.warning(results)
        return results

if __name__ == '__main__':

    app.run(host='0.0.0.0',port=8934)
