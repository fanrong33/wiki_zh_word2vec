# encoding: utf-8
# 将xml的wiki数据转换成text格式

from __future__ import print_function

import logging
import os.path
import sys

from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python process_wiki.py zhwiki.xxx.xml.bz2 wiki.zh.txt")
        sys.exit(1)
    
    input_file, output_file = sys.argv[1:3]
    i = 0

    fp = open(output_file, 'w')
    wiki = WikiCorpus(input_file, lemmatize=False, dictionary={}) # gensim里的维基百科处理类WikiCorpus
    for text in wiki.get_texts(): # 通过get_texts将维基里的每篇文章转换为1行text文本，并且去掉了标点符号等内容
        fp.write(' '.join(text)+'\n')
            
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    fp.close()
    logger.info("Finished Saved " + str(i) + " articles")