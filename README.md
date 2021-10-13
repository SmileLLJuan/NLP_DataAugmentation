# NLP_DataAugmentation
数据增强:nlp中生成文本的相似表达
现阶段支持文本生成方式：
1、同义词替换
主要思想：随机替换文本中词语的同义词

调用方式：

`from codes.word_replace import SimilarWordReplace

vocab = "../data/similar_words.txt"

wordreplace = SimilarWordReplace(vocab)

rep = wordreplace.run("我们一起去打羽毛球吧")

print(rep)  # 我们一同去打羽毛球吧`

2、中英回译
主要思想：中文文本翻译成英文，再由英文翻译成中文

调用方式：

`from codes.back_translation import sentence_trans

sentence = "你最喜欢的人是我吗？"  

resu = sentence_trans(sentence)  

print(resu)`

3、基于simbert的方法
参考文献：https://spaces.ac.cn/archives/8454
主要思想：利用uniLM思想训练seq2seq模型，学习模型NLG能力；利用transformers双向encoder网络学习文本的NLU能力；

调用方式：

`from codes.text_Synonyms import SynonymsGenerator

model_path="/mnt/disk1/lilijuan/corpus/trained_bert_models/chinese_simbert_L-12_H-768_A-12"

synonyms=SynonymsGenerator(model_path=model_path)

synonyms_result = synonyms.gen_synonyms(texts="怎么开初婚未育证明", n=20, threhold=0.9)

print(synonyms_result)`