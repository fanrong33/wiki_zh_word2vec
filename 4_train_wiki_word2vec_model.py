# encoding: utf-8
# 使用gensim word2vec训练获取wiki百科词向量

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence




# inp为输入语料，outp1 为输出模型，outp2 为原始c版本word2vec的vector格式的模式
inp = 'wiki.zh.simp.seg.txt'
outp1 = 'wiki.zh.text.model'
outp2 = 'wiki.zh.text.vector'
model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=4)
model.save(outp1)
model.wv.save_word2vec_format(outp2, binary=False)