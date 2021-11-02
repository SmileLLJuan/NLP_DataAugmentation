#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/26 10:22
# @Author  : lilijuan
# @File    : text_Synonyms.py

import os,json,re
import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
import tensorflow as tf

def setup_seed(seed):
    try:
        import random
        import numpy as np
        np.random.seed(seed)
        random.seed(seed)
    except Exception as e:
        pass


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    def __init__(self, model_path, max_len=32, seed=1):
        # super().__init__()
        keras.backend.clear_session()
        self.graph = tf.compat.v1.get_default_graph()  # not in the same graph() 问题
        setup_seed(seed)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        with self.graph.as_default():
            with self.session.as_default():   #解决keras web服务not in the same grap问题
                self.config_path = os.path.join(model_path, "bert_config.json")
                self.checkpoint_path = os.path.join(model_path, "bert_model.ckpt")
                self.dict_path = os.path.join(model_path, "vocab.txt")
                self.max_len = max_len
                self.tokenizer = Tokenizer(self.dict_path, do_lower_case=True)
                self.bert = build_transformer_model(
                    self.config_path,
                    self.checkpoint_path,
                    model='roformer' if "roformer" in model_path else "bert",  # SimBERTv2模型加载, SimBERT模型加载时, 注释该行
                    with_pool='linear',
                    application='unilm',
                    return_keras_model=False,
                )
                self.encoder = keras.models.Model(self.bert.model.inputs,
                                                  self.bert.model.outputs[0])
                self.seq2seq = keras.models.Model(self.bert.model.inputs,
                                                  self.bert.model.outputs[1])
                x, s = self.tokenizer.encode("你好")
                Z = self.encoder.predict([[x], [s]])   #[batch_size,768]
                Z = self.seq2seq.predict([[x], [s]])   #[batch_size,seq_length+2,12000]
                super().__init__(start_id=None, end_id=self.tokenizer._token_end_id,
                             maxlen=self.max_len)

# @AutoRegressiveDecoder.set_rtype('probas')  # bert4keras==0.7.7
    @AutoRegressiveDecoder.wraps(default_rtype='probas')  # bert4keras==0.10.6
    def predict(self, inputs, output_ids, states):
        # with self.graph.as_default():
        #     with self.session.as_default():
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        # bert4keras==0.7.7
        # token_ids, segment_ids = self.tokenizer.encode(
        #     text, max_length=self.max_len)

        # bert4keras==0.10.6

        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_len)
        output_ids = self.random_sample([token_ids, segment_ids], n, topk)
        return [self.tokenizer.decode(ids) for ids in output_ids]

    def gen_text_synonyms(self, text, n=100, k=20, threhold=0.75,distinguish_symbol=True):
        """"含义： 产生sent的n个相似句，然后返回最相似的k个。
        做法：用seq2seq生成，并用encoder算相似度并排序。
        """
        with self.graph.as_default():
            with self.session.as_default():
                text=text.replace("\n","")
                r = self.generate(text, n)
                r = [i for i in set(r) if i != text]
                r = [text] + r
                X, S = [], []
                for t in r:
                    x, s = self.tokenizer.encode(t)
                    X.append(x)
                    S.append(s)
                X = sequence_padding(X)
                S = sequence_padding(S)
                Z = self.encoder.predict([X, S])
                Z /= (Z ** 2).sum(axis=1, keepdims=True) ** 0.5
                scores = np.dot(Z[1:], Z[0])
                argsort = scores.argsort()
                scores = scores.tolist()

        if distinguish_symbol:
            # 只添加了标点符号的案例不添加进相似问题
            flag_same_char_len=lambda x,y:re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，：:。？?、~@#￥%……&*（）]+", "",x)==re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，：:。？?、~@#￥%……&*（）]+", "",y)
            return {"text":text,"synonyms":[(r[i + 1], scores[i]) for i in argsort[::-1][:k] if scores[i] > threhold and flag_same_char_len(text,r[i+1])==False]}
        else:
            return {"text":text,"synonyms":[(r[i + 1], scores[i]) for i in argsort[::-1][:k] if scores[i] > threhold]}

    def gen_texts_synonyms(self, texts, n=100, k=20, threhold=0.75):
        results={}
        for i,text in enumerate(texts):
            synonyms_list=self.gen_text_synonyms(text,n,k,threhold)
            results[i]={"text":text,"synonyms":synonyms_list['synonyms']}
        return results

    def gen_file_synonyms(self, file, n=100, k=20, threhold=0.75):
        with open(file,'r',encoding='utf-8') as f:
            texts=f.readlines()
        results=self.gen_texts_synonyms(texts,n,k,threhold)
        file_name = file.split("/")[-1]
        save_file_name = file_name.split('.')[0] + "_results" + "." + file_name.split('.')[-1] if len(
            file_name.split('.')) > 1 else file_name.split('.')[0] + "_results"
        file_save = "/".join(file.split("/")[:-1]) + "/" + save_file_name
        with open(file_save,'w',encoding='utf-8') as f:
            for i,r in results.items():
                f.write(r['text']+"\t"+'1.0'+"\n")
                for t in r['synonyms']:
                    f.write(t[0]+"\t"+str(t[1])+"\n")
        return {"file":file,"save_file":file_save}
    def gen_synonyms(self,texts, n=100, k=20, threhold=0.75):
        results = {}
        if type(texts)==str and os.path.exists(texts)==True:
            results=self.gen_file_synonyms(texts, n=n, k=k, threhold=threhold)
        elif type(texts)==str:
            results = self.gen_text_synonyms(texts, n, k, threhold)
        elif type(texts)==list:
            results = self.gen_texts_synonyms(texts, n, k, threhold)
        return results

if __name__ == '__main__':
    model_path="/mnt/disk1/lilijuan/corpus/trained_bert_models/chinese_roformer-sim-char_L-6_H-384_A-6"
    model_path="/mnt/disk1/lilijuan/corpus/trained_bert_models/chinese_roformer-sim-char_L-12_H-768_A-12"
    # model_path="/mnt/disk1/lilijuan/corpus/trained_bert_models/chinese_simbert_L-12_H-768_A-12"
    synonyms=SynonymsGenerator(model_path=model_path)
    # synonyms_result = synonyms.gen_file_synonyms(file="/mnt/disk1/lilijuan/HK/SimBERT/data/file_sysnomys_test", n=10, threhold=0.9)
    # print(synonyms_result)
    synonyms_result = synonyms.gen_synonyms(texts="怎么开初婚未育证明", n=30, threhold=0.9)
    print(synonyms_result)
    texts=['怎么开初婚未育证明','帮我关一下台灯','我想吃附近的火锅','我们一起去打羽毛球吧','把我的一个亿存银行安全吗','微信和支付宝哪个好？',]
    for text in texts:
        text = text.replace("\n", "")
        if len(text) > 2:
            synonyms_result= synonyms.gen_synonyms(texts=text, n=10, threhold=0.9)
            print(text, synonyms_result,"#"*20)
            for t,s in synonyms_result['synonyms']:
                print(t,s)

