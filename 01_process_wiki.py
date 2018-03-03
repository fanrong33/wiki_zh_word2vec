# encoding: utf-8
""" 将xml的wiki数据转换成text格式

@version 1.0.1 build 20180303
"""

from __future__ import print_function

import logging
import sys
import os.path
from gensim.corpora import WikiCorpus


if __name__ == '__main__':
    logger = logging.getLogger()

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)


    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python process_wiki.py zhwiki.xxx.xml.bz2 wiki.zh.txt")
        sys.exit(1)
    

    # 从命令行获取参数
    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    i = 0

    with open(output_file, 'w') as fp:
        wiki = WikiCorpus(input_file, lemmatize=False, dictionary={}) # gensim里的维基百科处理类WikiCorpus
        for text in wiki.get_texts(): # 通过get_texts将维基里的每篇文章转换为1行text文本，并且去掉了标点符号等内容
            fp.write(' '.join(text)+'\n')

            i = i + 1
            if (i % 10000 == 0):
                logger.info("Saved " + str(i) + " articles")

        logger.info("Finished Saved " + str(i) + " articles")


'''
2017-04-18 09:24:28,901: INFO: running 1_process.py zhwiki-latest-pages-articles.xml.bz2 wiki.zh.txt
2017-04-18 09:25:31,154: INFO: Saved 10000 articles.
2017-04-18 09:26:21,582: INFO: Saved 20000 articles.
2017-04-18 09:27:05,642: INFO: Saved 30000 articles.
2017-04-18 09:27:48,917: INFO: Saved 40000 articles.
2017-04-18 09:28:35,546: INFO: Saved 50000 articles.
2017-04-18 09:29:21,102: INFO: Saved 60000 articles.
2017-04-18 09:30:04,540: INFO: Saved 70000 articles.
2017-04-18 09:30:48,022: INFO: Saved 80000 articles.
2017-04-18 09:31:30,665: INFO: Saved 90000 articles.
2017-04-18 09:32:17,599: INFO: Saved 100000 articles.
2017-04-18 09:33:13,811: INFO: Saved 110000 articles.
2017-04-18 09:34:06,316: INFO: Saved 120000 articles.
2017-04-18 09:35:01,007: INFO: Saved 130000 articles.
2017-04-18 09:35:52,628: INFO: Saved 140000 articles.
2017-04-18 09:36:47,148: INFO: Saved 150000 articles.
2017-04-18 09:37:41,137: INFO: Saved 160000 articles.
2017-04-18 09:38:33,684: INFO: Saved 170000 articles.
2017-04-18 09:39:37,957: INFO: Saved 180000 articles.
2017-04-18 09:43:36,299: INFO: Saved 190000 articles.
2017-04-18 09:45:21,509: INFO: Saved 200000 articles.
2017-04-18 09:46:40,865: INFO: Saved 210000 articles.
2017-04-18 09:47:55,453: INFO: Saved 220000 articles.
2017-04-18 09:49:07,835: INFO: Saved 230000 articles.
2017-04-18 09:50:27,562: INFO: Saved 240000 articles.
2017-04-18 09:51:38,755: INFO: Saved 250000 articles.
2017-04-18 09:52:50,240: INFO: Saved 260000 articles.
2017-04-18 09:53:57,526: INFO: Saved 270000 articles.
2017-04-18 09:55:01,720: INFO: Saved 280000 articles.
2017-04-18 09:55:22,565: INFO: finished iterating over Wikipedia corpus of 28285 5 documents with 63427579 positions (total 2908316 articles, 75814559 positions before pruning articles shorter than 50 words)
2017-04-18 09:55:22,568: INFO: Finished Saved 282855 articles.
'''

