# encoding: utf-8
# [models.word2vec – Deep learning with word2vec](https://radimrehurek.com/gensim/models/word2vec.html)

from gensim.models import Word2Vec
import logging
import time


logger = logging.getLogger()

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


start = time.time()

model = Word2Vec.load('wiki.zh.text.model')

cost_time = time.time() - start
logger.info("Loading model cost %.3f seconds" % cost_time)
# 2018-03-03 13:44:10,474: DEBUG: Loading model cost 0.896 seconds.


# 与“足球”相似的词语
results = model.wv.most_similar('足球')
# print(results)
'''
[('国际足球', 0.5326899886131287), ('篮球', 0.5310245156288147), ('足球运动', 0.5240993499755859), ('足球队', 0.5158734321594238), ('冰球', 0.5030774474143982), ('足球联赛', 0.4926849901676178), ('体育', 0.4907689690589905), ('排球', 0.48957353830337524), ('国家足球队', 0.4877948760986328), ('男子篮球', 0.481542706489563)]
'''
results = model.wv.most_similar('暴涨')

# 男人 -> 爸爸， 女人 -> ?
results = model.wv.most_similar(positive=['女人','爸爸'], negative=['男人'])
# print(results)
'''
[('妈妈', 0.5810824036598206), ('奶奶', 0.5117344856262207), ('太太', 0.5021586418151855), ('老公', 0.4986879825592041), ('老婆', 0.49133044481277466), ('母亲', 0.4811268448829651), ('老爸', 0.47127312421798706), ('女儿', 0.46257784962654114), ('外婆', 0.4621780514717102), ('爷爷', 0.4615582227706909)]
'''
# 中国 -> 上海， 美国 -> ?
results = model.wv.most_similar(positive=['美国', '上海'], negative=['中国'])



# 计算词1和词2的余弦相似度
sim = model.wv.similarity('书籍', '书本')
# print(sim)
''' 0.587002285182 '''
sim = model.wv.similarity('逛街', '书本')
# print(sim)
''' 0.17013282452 '''

sim = model.wv.similarity(u'男朋友', u'女朋友')
# print(sim)
''' 0.830840540457 '''
if '学弟' in model.wv.vocab:
    sim = model.wv.similarity(u'学弟', u'学长')
    # print(sim)
    ''' 0.803594008944 '''



# 计算两个数据集间的余弦相似度, 可以用于实现相似文章功能
sim = model.wv.n_similarity(['苹果','手机'], ['安卓','手机'])
# print(sim)
''' 0.846801993317 '''
sim = model.wv.n_similarity(['苹果','手机'], ['天气','预报'])
# print(sim)
''' 0.201779894517 '''

# 计算两个集合之间的余弦相似度
sim1 = model.n_similarity(['得到', '拿下'], ['送出', '送出'])
# print(sim1)
''' 0.333287002201 '''


# 数据集中不匹配的一项
word = model.wv.doesnt_match(['太后','妃子','贵人','贵妃','嫔妃'])
# print(word)
''' 太后 '''






'''
2018-03-03 16:15:14,693: INFO: loading Word2Vec object from wiki.zh.text.model
2018-03-03 16:15:17,473: INFO: loading wv recursively from wiki.zh.text.model.wv.* with mmap=None
2018-03-03 16:15:17,473: INFO: loading vectors from wiki.zh.text.model.wv.vectors.npy with mmap=None
2018-03-03 16:15:18,926: INFO: setting ignored attribute vectors_norm to None
2018-03-03 16:15:18,926: INFO: loading vocabulary recursively from wiki.zh.text.model.vocabulary.* with mmap=None
2018-03-03 16:15:18,926: INFO: loading trainables recursively from wiki.zh.text.model.trainables.* with mmap=None
2018-03-03 16:15:18,927: INFO: loading syn1neg from wiki.zh.text.model.trainables.syn1neg.npy with mmap=None
2018-03-03 16:15:20,809: INFO: setting ignored attribute cum_table to None
2018-03-03 16:15:20,809: INFO: loaded wiki.zh.text.model
2018-03-03 16:15:22,567: INFO: Loading model cost 7.875 seconds
2018-03-03 16:15:22,568: INFO: precomputing L2-norms of word weight vectors
'''


