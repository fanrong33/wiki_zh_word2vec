# encoding: utf-8
# 逐行读取wiki数据去除英文，并进行分词
# 由于此语料已经去除了标点符号，
# 因此在分词程序中无需进行清洗操作，可直接分词。
# 若是自己采集的数据还需进行标点符号去除和去除停用词的操作。

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
    with codecs.open('wiki.zh.simp.seg.txt', "a+",'utf-8') as fp:
        for line in open("wiki.zh.simp.txt"):

            for line2 in re.sub('([a-zA-Z0-9]|é)', '', line).split(' '):
                if line2 != '':
                    seg_list = list(jieba.cut(line2, cut_all=False))
                    line_seg=' '.join(seg_list) # +'\n'
                    fp.writelines(line_seg)
            line_num = line_num + 1
            if (line_num % 10000 == 0):
                # print("Saved " + str(line_num) + " articles")
                logger.info("Saved " + str(line_num) + " articles")

    logger.info("Finished Saved " + str(line_num) + " articles")


# TODO 默认jieba配置，待优化暂停词，自定义词典
# áó