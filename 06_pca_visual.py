# encoding: utf-8
# [models.word2vec – Deep learning with word2vec](https://radimrehurek.com/gensim/models/word2vec.html)

from gensim.models import Word2Vec
from sklearn.decomposition import PCA # decomposition 分解
from matplotlib import pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


model = Word2Vec.load('wiki.zh.text.model')

# X = model[model.wv.vocab]

# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# plt.scatter(result[:, 0], result[:, 1])
# plt.show()


word = ["爸爸", "妈妈"]
# 寻找出最相似的多个词
words = [wp[0] for wp in model.most_similar(word, topn=20)]
# print(words)
'''
['老公', '奶奶', '爷爷', '阿姨', '老婆', '老爸', '母亲', '保姆', '大叔', '哥哥', '外婆', '爸妈', '姊姊', '妈咪', '婆婆', '太太', '妹妹', '小宝宝', '小兔', '女儿']
'''

# 提取出词对应的词向量
words_in_vector = [model[word] for word in words]
# print(words_in_vector)

# print(model['爸爸'])
''' size长度为400的词向量
[  1.22842872e+00   9.84687269e-01   9.24556017e-01  -2.57590771e-01
   ...
   1.67887378e+00  -2.94714928e+00  -1.82157099e+00  -4.83914346e-01]
'''

# 训练 PCA 模型进行降维
pca = PCA(n_components=2)  # 只保留2个维度
pca.fit(words_in_vector)

X = pca.transform(words_in_vector)

# 绘制图形
xs = X[:, 0]
ys = X[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(xs, ys, marker='o')

# 遍历所有的词添加点注释
for i, w in enumerate(words):
    plt.annotate(
        w,
        xy=(xs[i], ys[i]), xytext=(6, 6),
        textcoords='offset points', ha='left', va='top',
        **dict(fontsize=10)
    )
plt.show()


