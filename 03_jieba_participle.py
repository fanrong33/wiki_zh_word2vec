# encoding: utf-8
""" 结巴分词，逐行读取wiki数据去除英文，并进行分词

由于此语料已经去除了标点符号，
因此在分词程序中无需进行清洗操作，可直接分词。
若是自己采集的数据还需进行标点符号去除和去除停用词的操作。

注意：英文的分词则只需要 split()

@version 1.0.1 build 20180303
"""

import jieba
import jieba.analyse
import jieba.posseg # 引入词性标注接口
import codecs
import re

import logging
import os.path
import sys


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))


    # 将wiki中每篇文章的每个句子分词后按行存储
    line_num = 0
    with codecs.open('wiki.zh.simp.seg.txt', "w",'utf8') as fp:
        with codecs.open("wiki.zh.simp.txt", 'r', 'utf8') as input_fp:
            for line in input_fp.readlines():
                seg_list = list(jieba.cut(line, cut_all=False))
                line_seg =' '.join(seg_list) # +'\n'
                fp.writelines(line_seg)
                line_num = line_num + 1
            if (line_num % 10000 == 0):
                # print("Saved " + str(line_num) + " articles")
                logger.info("Saved " + str(line_num) + " articles")

    logger.info("Finished Saved " + str(line_num) + " articles")


# TODO 默认jieba配置，待优化暂停词，自定义词典

'''
Building prefix dict from the default dictionary ...
2018-03-03 13:44:09,578: DEBUG: Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/dl/z5zg2s6x4p1frvxqt_ddwk7h0000gn/T/jieba.cache
2018-03-03 13:44:09,579: DEBUG: Loading model from cache /var/folders/dl/z5zg2s6x4p1frvxqt_ddwk7h0000gn/T/jieba.cache
Loading model cost 0.896 seconds.
2018-03-03 13:44:10,474: DEBUG: Loading model cost 0.896 seconds.
Prefix dict has been built succesfully.
2018-03-03 13:44:10,474: DEBUG: Prefix dict has been built succesfully.
2018-03-03 14:30:36,885: INFO: Finished Saved 306129 articles
'''


