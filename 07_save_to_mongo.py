# encoding: utf-8
""" 将词向量模型存储到mongodb服务, 使用批量insert
@version 1.0.2 build 20180412
"""

import logging

from pymongo import MongoClient
from gensim.models import Word2Vec

from bson.binary import Binary
import pickle

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


# 连接mongodb
conn = MongoClient('mongodb://root:root@127.0.0.1:27017')
db = conn.word2vec


model = Word2Vec.load('wiki.zh.text.model')
total = len(model.wv.vocab)
logger.info('vocab total %d.' % total)


# 6
# 0 1 2 3 4 5

# 1 < 5  0 < 5
# 2 < 5  1 < 5
# 3 < 5  2 < 5
# 4 < 5  3 < 5
# 5 < 5  4 < 5 insert

# 1 < 5  5 < 5 insert

array = []
for i, word in enumerate(model.wv.vocab):
    # if i <= 773547-1:
    #     continue

    batch_size = 10

    nparray = model.wv[word]
    data = {'word': word,'vec_binary': Binary(pickle.dumps(nparray, protocol=-1))}
    array.append(data)

    if len(array) < batch_size and i < total-1:
        continue

    db.word2vec.insert(array)
    array = []  # reset array

    if i % 100 == 0:
        logger.info('insert %d words' % i)


