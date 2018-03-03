# encoding: utf-8
# 使用gensim word2vec训练获取wiki百科词向量

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import logging
import os.path
import sys


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)


    logger.info("running %s" % ' '.join(sys.argv))


    # input_text为输入语料，output1 为输出模型，output2 为原始c版本word2vec的vector格式的模式
    input_text = 'wiki.zh.simp.seg.txt'
    output1    = 'wiki.zh.text.model'  # 模型
    output2    = 'wiki.zh.text.vector' # 是每个词对应的词向量，可以在此基础上作文本特征的提取以及分类

    # 训练skip-gram 模型
    # sentences 是句子序列，句子又是单词列表，
    #   如 [['蒙牛','牛奶','好喝'],['三星','手机'],['名车','抢眼','奥迪','拍照']]
    # min_count 表示小于该数的单词会被剔除，默认值为5
    # size 表示神经网络的隐藏层单元数，默认为100
    model = Word2Vec(LineSentence(input_text), size=400, window=5, min_count=5, workers=4)

    # 保存模型
    model.save(output1)
    model.wv.save_word2vec_format(output2, binary=False) 



""" 训练和保存模型耗时接近50分钟
2018-03-03 14:44:58,957: INFO: running 04_train_wiki_word2vec_model.py
2018-03-03 14:44:58,957: INFO: collecting all words and their counts
2018-03-03 14:44:58,958: INFO: PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
2018-03-03 14:45:06,237: INFO: PROGRESS: at sentence #10000, processed 12076465 words, keeping 584534 word types
2018-03-03 14:45:11,735: INFO: PROGRESS: at sentence #20000, processed 20819895 words, keeping 840225 word types
2018-03-03 14:45:15,493: INFO: PROGRESS: at sentence #30000, processed 28681724 words, keeping 1019174 word types
2018-03-03 14:45:19,157: INFO: PROGRESS: at sentence #40000, processed 35985805 words, keeping 1184744 word types
2018-03-03 14:45:23,199: INFO: PROGRESS: at sentence #50000, processed 42903877 words, keeping 1329127 word types
2018-03-03 14:45:26,450: INFO: PROGRESS: at sentence #60000, processed 49397697 words, keeping 1452220 word types
2018-03-03 14:45:29,534: INFO: PROGRESS: at sentence #70000, processed 55545175 words, keeping 1565347 word types
2018-03-03 14:45:33,187: INFO: PROGRESS: at sentence #80000, processed 61562702 words, keeping 1679433 word types
2018-03-03 14:45:36,457: INFO: PROGRESS: at sentence #90000, processed 67346994 words, keeping 1781922 word types
2018-03-03 14:45:39,454: INFO: PROGRESS: at sentence #100000, processed 73282183 words, keeping 1879416 word types
2018-03-03 14:45:42,491: INFO: PROGRESS: at sentence #110000, processed 78859328 words, keeping 1964493 word types
2018-03-03 14:45:45,208: INFO: PROGRESS: at sentence #120000, processed 83957081 words, keeping 2046111 word types
2018-03-03 14:45:48,164: INFO: PROGRESS: at sentence #130000, processed 89753809 words, keeping 2136436 word types
2018-03-03 14:45:51,062: INFO: PROGRESS: at sentence #140000, processed 95184182 words, keeping 2215580 word types
2018-03-03 14:45:53,833: INFO: PROGRESS: at sentence #150000, processed 100752389 words, keeping 2296620 word types
2018-03-03 14:45:56,767: INFO: PROGRESS: at sentence #160000, processed 106157290 words, keeping 2379678 word types
2018-03-03 14:45:59,568: INFO: PROGRESS: at sentence #170000, processed 111759321 words, keeping 2449273 word types
2018-03-03 14:46:03,110: INFO: PROGRESS: at sentence #180000, processed 116822487 words, keeping 2519951 word types
2018-03-03 14:46:05,817: INFO: PROGRESS: at sentence #190000, processed 121355957 words, keeping 2588742 word types
2018-03-03 14:46:08,730: INFO: PROGRESS: at sentence #200000, processed 126251730 words, keeping 2658752 word types
2018-03-03 14:46:11,492: INFO: PROGRESS: at sentence #210000, processed 131274725 words, keeping 2712203 word types
2018-03-03 14:46:14,562: INFO: PROGRESS: at sentence #220000, processed 136446473 words, keeping 2777945 word types
2018-03-03 14:46:18,482: INFO: PROGRESS: at sentence #230000, processed 141635361 words, keeping 2839266 word types
2018-03-03 14:46:21,367: INFO: PROGRESS: at sentence #240000, processed 146620880 words, keeping 2893507 word types
2018-03-03 14:46:24,583: INFO: PROGRESS: at sentence #250000, processed 152047972 words, keeping 2957741 word types
2018-03-03 14:46:27,933: INFO: PROGRESS: at sentence #260000, processed 157445118 words, keeping 3018583 word types
2018-03-03 14:46:31,316: INFO: PROGRESS: at sentence #270000, processed 162733707 words, keeping 3077231 word types
2018-03-03 14:46:34,304: INFO: PROGRESS: at sentence #280000, processed 167379722 words, keeping 3130859 word types
2018-03-03 14:46:37,031: INFO: PROGRESS: at sentence #290000, processed 171836650 words, keeping 3179843 word types
2018-03-03 14:46:39,711: INFO: PROGRESS: at sentence #300000, processed 176147397 words, keeping 3229235 word types
2018-03-03 14:46:41,643: INFO: collected 3265764 word types from a corpus of 179479398 raw words and 306572 sentences
2018-03-03 14:46:41,643: INFO: Loading a fresh vocabulary
2018-03-03 14:46:46,031: INFO: min_count=5 retains 774118 unique words (23% of original 3265764, drops 2491646)
2018-03-03 14:46:46,031: INFO: min_count=5 leaves 175685280 word corpus (97% of original 179479398, drops 3794118)
2018-03-03 14:46:48,609: INFO: deleting the raw counts dictionary of 3265764 items
2018-03-03 14:46:48,747: INFO: sample=0.001 downsamples 15 most-common words
2018-03-03 14:46:48,747: INFO: downsampling leaves estimated 167211503 word corpus (95.2% of prior 175685280)
2018-03-03 14:46:52,117: INFO: estimated required memory for 774118 words and 400 dimensions: 2864236600 bytes
2018-03-03 14:46:52,117: INFO: resetting layer weights
2018-03-03 14:47:09,928: INFO: training model with 4 workers on 774118 vocabulary and 400 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
2018-03-03 14:47:10,985: INFO: EPOCH 1 - PROGRESS: at 0.01% examples, 39824 words/s, in_qsize 7, out_qsize 0
2018-03-03 14:49:00,024: INFO: EPOCH 1 - PROGRESS: at 10.09% examples, 246940 words/s, in_qsize 8, out_qsize 0
2018-03-03 14:50:19,650: INFO: EPOCH 1 - PROGRESS: at 20.09% examples, 246101 words/s, in_qsize 7, out_qsize 0
2018-03-03 14:51:28,367: INFO: EPOCH 1 - PROGRESS: at 30.08% examples, 246278 words/s, in_qsize 7, out_qsize 1
2018-03-03 14:52:16,233: INFO: EPOCH 1 - PROGRESS: at 40.13% examples, 259685 words/s, in_qsize 7, out_qsize 0
2018-03-03 14:53:04,263: INFO: EPOCH 1 - PROGRESS: at 50.08% examples, 269102 words/s, in_qsize 6, out_qsize 1
2018-03-03 14:53:49,338: INFO: EPOCH 1 - PROGRESS: at 60.26% examples, 276934 words/s, in_qsize 7, out_qsize 0
2018-03-03 14:54:30,335: INFO: EPOCH 1 - PROGRESS: at 70.02% examples, 282420 words/s, in_qsize 6, out_qsize 1
2018-03-03 14:55:13,311: INFO: EPOCH 1 - PROGRESS: at 80.10% examples, 288274 words/s, in_qsize 7, out_qsize 0
2018-03-03 14:55:56,323: INFO: EPOCH 1 - PROGRESS: at 90.06% examples, 292973 words/s, in_qsize 7, out_qsize 0
2018-03-03 14:56:34,264: INFO: EPOCH 1 - PROGRESS: at 99.85% examples, 295844 words/s, in_qsize 7, out_qsize 0
2018-03-03 14:56:35,244: INFO: worker thread finished; awaiting finish of 3 more threads
2018-03-03 14:56:35,258: INFO: worker thread finished; awaiting finish of 2 more threads
2018-03-03 14:56:35,281: INFO: EPOCH 1 - PROGRESS: at 99.99% examples, 295767 words/s, in_qsize 1, out_qsize 1
2018-03-03 14:56:35,281: INFO: worker thread finished; awaiting finish of 1 more threads
2018-03-03 14:56:35,284: INFO: worker thread finished; awaiting finish of 0 more threads
2018-03-03 14:56:35,284: INFO: EPOCH - 1 : training on 179479398 raw words (167209815 effective words) took 565.3s, 295778 effective words/s
2018-03-03 15:29:26,027: INFO: EPOCH 5 - PROGRESS: at 99.97% examples, 398408 words/s, in_qsize 3, out_qsize 1
2018-03-03 15:29:26,027: INFO: worker thread finished; awaiting finish of 3 more threads
2018-03-03 15:29:26,051: INFO: worker thread finished; awaiting finish of 2 more threads
2018-03-03 15:29:26,054: INFO: worker thread finished; awaiting finish of 1 more threads
2018-03-03 15:29:26,055: INFO: worker thread finished; awaiting finish of 0 more threads
2018-03-03 15:29:26,055: INFO: EPOCH - 5 : training on 179479398 raw words (167211657 effective words) took 419.7s, 398439 effective words/s
2018-03-03 15:29:26,056: INFO: training on a 897396990 raw words (836057002 effective words) took 2536.0s, 329670 effective words/s
2018-03-03 15:29:26,080: INFO: saving Word2Vec object under wiki.zh.text.model, separately None
2018-03-03 15:29:26,095: INFO: storing np array 'vectors' to wiki.zh.text.model.wv.vectors.npy
2018-03-03 15:29:30,918: INFO: not storing attribute vectors_norm
2018-03-03 15:29:30,922: INFO: storing np array 'syn1neg' to wiki.zh.text.model.trainables.syn1neg.npy
2018-03-03 15:29:37,101: INFO: not storing attribute cum_table
2018-03-03 15:29:39,778: INFO: saved wiki.zh.text.model
2018-03-03 15:29:39,788: INFO: storing 774118x400 projection weights into wiki.zh.text.vector
"""



