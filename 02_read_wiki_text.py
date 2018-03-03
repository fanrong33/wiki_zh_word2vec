# encoding: utf-8
""" 查看转换后的简体维基百科数据，因为编辑器无法打开，所以采用python自带的IO进行读取

@version 1.0.1 build 20180303
"""

import codecs
import sys

f = codecs.open('wiki.zh.simp.txt', 'r', encoding='utf-8')
line = f.readline()
print(line)
'''
欧几里得 西元前三世纪的希腊数学家 现在被认为是几何之父 此画为拉斐尔的作品 雅典学院 数学 是利用符号语言研究数量 结构 变化以及空间等概念的一门学科 从某种角度看属于形式科学的一种 数学透过抽象化和逻辑推理的使用 由计数 计算 数学家们拓展这些概念 对数学基本概念的完善 早在古埃及 而在古希腊那里有更为严谨的处理 从那时开始 数学的发展便持续不断地小幅进展 世纪的文艺复兴时期 致使数学的加速发展 直至今日 今日 数学使用在不同的领域中 包括科学 工程 医学和经济学等 有时亦会激起新的数学发现 并导致全新学科的发展 数学家也研究纯数学 就是数学本身的实质性内容 而不以任何实际应用为目标 虽然许多研究以纯数学开始 但其过程中也发现许多应用之处 词源 西方语言中 数学 一词源自于古希腊语的 其有 学习 学问 科学 数学研究 即使在其语源内 其形容词 意思为 和学习有关的 用功的 亦会被用来指 数学的 其在英语中表面上的复数形式 及在法语中的表面复数形式 可溯至拉丁文的中性复数 由西塞罗译自希腊文复数 此一希腊语被亚里士多德拿来指 万物皆数 的概念 汉字表示的 数学 一词大约产生于中国宋元时期 多指象数之学 但有时也含有今天上的数学意义 例如 秦九韶的 数学九章 永乐大典 数书九章 也被宋代周密所著的 癸辛杂识 记为 数学大略 数学通轨 明代柯尚迁著 数学钥 清代杜知耕著 数学拾遗 清代丁取忠撰 直到 经过中国数学名词审查委员会研究 算学 数学 两词的使用状况后 确认以 数学 表示今天意义上的数学含义 历史 奇普 印加帝国时所使用的计数工具 玛雅数字 数学有着久远的历史 中国古代的六艺之一就有 数学一词在西方有希腊语词源 mathematikós 意思是 学问的基础 源于 máthema 科学 知识 学问 时间的长短等抽象的数量关系 比如时间单位有日 季节和年等 算术 加减乘除 也自然而然地产生了 历史上曾有过许多不同的记数系统 在最初有历史记录的时候 为了解数字间的关系 为了测量土地 以及为了预测天文事件而形成的 结构 空间及时间方面的研究 到了 世纪 算术 微积分的概念也在此时形成 随着数学转向形式化 从古至今 数学便一直不断地延展 且与科学有丰富的相互作用 两者的发展都受惠于彼此 在历史上有著许多数学发现 并且直至今日都不断地有新的发现 据mikhail sevryuk于 月的期刊中所说 数学评论的创刊年份 现已超过了一百九十万份 而且每年还增加超过七万五千份 形成 纯数学与应用数学及美学 牛顿 微积分的发明者之一 每当有涉及数量 结构 空间及变化等方面的困难问题时 而这往往也拓展了数学的研究范畴 一开始 数学的运用可见于贸易 土地测量及之后的天文学 今日 且数学本身亦给出了许多的问题 牛顿和莱布尼兹是微积分的发明者 费曼发明了费曼路径积分 这是推理及物理洞察二者的产物 而今日的弦理论亦引申出新的数学 一些数学只和生成它的领域有关 且用来解答此领域的更多问题 且可以成为一般的数学概念 即使是 最纯的 数学通常亦有实际的用途 此一非比寻常的事实 年诺贝尔物理奖得主维格纳称为 如同大多数的研究领域 主要的分歧为纯数学和应用数学 在应用数学内 又被分成两大领域 并且变成了它们自身的学科 统计学和电脑科学 许多数学家谈论数学的 优美 其内在的美学及美 简单 一般化 即为美的一种 另外亦包括巧妙的证明 又或者是加快计算的数值方法 如快速傅立叶变换 高德菲 哈罗德 哈代在 一个数学家的自白 符号 语言与精确性 在现代的符号中 此一图像即产生自x cos arccos sin〡 arcsin cos〡 世纪后才被发明出来的 在此之前 数学以文字的形式书写出来 这种形式会限制了数学的发展 但初学者却常对此感到怯步 它被极度的压缩 少量的符号包含著大量的讯息 如同音乐符号一般 现今的数学符号有明确的语法 并且有效地对讯息作编码 这是其他书写方式难以做到的 符号化和形式化使得数学迅速发展 数学语言亦对初学者而言感到困难 亦困恼著初学者的 开放 等字在数学里有著特别的意思 数学术语亦包括如 同胚 可积性 等专有名词 数学需要比日常用语更多的精确性 严谨 但在现实应用中 定理 希腊人期许著仔细的论证 但在牛顿的时代 所使用的方法则较不严谨 牛顿为了解决问题所做的定义 今日 当大量的计算难以被验证时 其证明亦很难说是足够地严谨 公理在传统的思想中是 不证自明的真理 但这种想法是有问题的 在形式上 公理只是一串符号 但依据哥德尔不完备定理 尽管如此 在此意义下 数学作为科学 卡尔 弗里德里希 高斯 卡尔 弗里德里希 高斯称数学为 科学的皇后 在拉丁原文 以及其德语 对应于 科学 的单字的意思皆为知识 领域 而实际上 科学 若认为科学是只指物理的世界时 则数学 或至少是纯数学 不会是一门科学 爱因斯坦曾如此描述 数学定律越和现实有关 它们越不确定 若它们越是确定的话 它们和现实越不会有关 且因此不是卡尔 波普尔所定义的科学 但在 年代时 且波普尔推断 大部份的数学定律 如物理及生物学一样 是假设演绎的 比它现在看起来更接近 然而 其他的思想家 如较著名的拉卡托斯 另一观点则为某些科学领域 如理论物理 是其公理为尝试著符合现实的数学 而事实上 理论物理学家齐曼 john ziman 即认为科学是一种公众知识 因此亦包含著数学 在任何的情况下 减轻了数学不使用科学方法的缺点 在史蒂芬 沃尔夫勒姆 年的著作 一种新科学 中他提出 数学家对此的态度并不一致 且因此基本上是个哲学家 是低估了其美学方面的重要性 被创造 如艺术 或是 被发现 如科学 的争议 大学院系划分中常见 科学和数学系 实际上 但在细节上却会分开 数学的各领域 有如反映在中国算盘上的一般 如上所述 了解数字间的关系 测量土地及预测天文事件 这四种需要大致地与数量 结构 空间及变化 即算术 代数 几何及分析 等数学上广泛的子领域相关连著 除了上述主要的关注之外 至逻辑 至集合论 基础 至不同科学的经验上的数学 应用数学 及较近代的至不确定性的严格研究 基础与哲学 为了阐明数学基础 并研究此一架构的结果 就其本身而言 现代逻辑被分成递归论 模型论和证明论 千禧年大奖难题中的p style border px solid ddd text align center margin auto cellspacing px px 数学逻辑 集合论 范畴论 纯粹数学 数量 数量的研究起于数 孪生质数猜想及哥德巴赫猜想 当数系更进一步发展时 整数被视为有理数的子集 而有理数则包含于实数中 连续的量即是以实数来表示的 实数则可以被进一步广义化成复数 从自然数亦可以推广到超限数 它形式化了计数至无限的这一概念 另一个研究的领域为大小 阿列夫数 style border px solid ddd text align center margin auto cellspacing 自然数 整数 有理数 实数 复数 结构 这些物件的结构性质被探讨于群 zh cn zh tw 等抽象系统中 该些物件事实上也就是这样的系统 此为代数的领域 在此有一个很重要的概念 即广义化至向量空间的向量 它于线性代数中被研究 数量 结构及空间 即变化 纯粹数学 是研究抽象结构的理论 结构 布尔巴基学派认为 有三种基本的抽象结构 代数结构 序结构 偏序 全序 拓扑结构 邻域 极限 连通性 维数 style border px solid ddd text align center margin auto cellspacing px px px px 数论 群论 图论 序理论 空间 空间的研究源自于几何 尤其是欧几里得几何 三角学则结合了空间及数 且包含有著名的勾股定理 非欧几里得几何 及拓扑学 数和空间在解析几何 结合了数和空间的概念 亦有著拓扑群的研究 结合了结构与空间 李群被用来研究空间 结构及变化 在其许多分支中 并包含有存在已久的庞加莱猜想 以及有争议的四色定理 庞加莱猜想已在 年确认由俄罗斯数学家格里戈里 佩雷尔曼证明 而四色定理已在 年由凯尼斯 阿佩尔和沃夫冈 哈肯用电脑证明 而从来没有由人力来验证过 style border px solid ddd text align center margin auto cellspacing px px px px px px 几何 三角学 微分几何 拓扑学 zh cn 分形 zh tw 碎形 测度论 变化 而微积分更为研究变化的有利工具 函数诞生于此 做为描述一变化的量的核心概念 而复分析则为复数的等价领域 黎曼猜想 数学最基本的未决问题之一 便是以复分析来描述的 泛函分析注重在函数的 一般为无限维 空间上 而这在微分方程中被研究 px px px px px px 微积分 向量分析 微分方程 动力系统 混沌理论 复分析 离散数学 这包含有可计算理论 计算复杂性理论及资讯理论 这包含现知最有力的模型 图灵机 尽管电脑硬体的快速进步 最后 且因此有压缩及熵等概念 作为一相对较新的领域 离散数学有许多基本的未解问题 其中最有名的为p np问题 千禧年大奖难题之一 一般相信此问题的解答是否定的 style border px solid ddd text align center margin auto cellspacing px px px 组合数学 计算理论 密码学 图论 应用数学 工商业及其他领域上之现实问题 应用数学中的一重要领域为统计学 分析与预测 大部份的实验 而比较觉得是合作团体的一份子 数值分析研究有什么计算方法 file gravitation space source png 数学物理 file svg 数学流体力学 file composite trapezoidal rule illustration small svg 数值分析 file maximum boxed png 最佳化 file two red dice svg 概率论 file oldfaithful png 统计学 file market data index nya on utc png 计量金融 file arbitrary gametree solved svg zh tw 赛局理论 zh cn 博弈论 file front pareto svg 数理经济学 file signal transduction pathways zh cn svg 生物数学 file linear programming example graph zh png 作业研究 file simple feedback control loop svg 控制论 数学奖项 菲尔兹奖牌正面 数学奖通常和其他科学的奖项分开 数学上最有名的奖为菲尔兹奖 创立于 每四年颁奖一次 它通常被认为是数学的诺贝尔奖 创立于 两者都颁奖于特定的工作主题 著名的 个问题 称为希尔伯特的 个问题 年由德国数学家大卫 希尔伯特所提出 另一新的七个重要问题 称为千禧年大奖难题 发表于 而当中只有一个问题 黎曼猜想 和希尔伯特的问题重复 菲尔兹奖 每四年颁奖一次 颁给有卓越贡献的年轻数学家 每次最多四人得奖 得奖者须在该年元旦前未满四十岁 是年轻数学家可以获得的最大奖项 它是据加拿大数学家约翰 查尔斯 菲尔兹的要求设立的 菲尔兹奖被视为 数学 界的诺贝尔奖 沃尔夫奖 由沃尔夫基金会颁发 该基金会于 年在以色列创立 年开始颁奖 创始人里卡多 沃尔夫是外交家 实业家和慈善家 阿贝尔奖 每年颁发一次 为了纪念 年挪威著名数学家尼尔斯 亨利克 阿贝尔二百周年诞辰 挪威政府宣布将开始颁发此种奖金 奖金的数额大致同诺贝尔奖相近 年挪威政府拨款 亿挪威克朗作为启动资金 扩大数学的影响 参见 数学哲学 数学游戏 数学家列表 教育 算经十书 数学竞赛 数学题 注记 参考书目 benson donald the moment of proof mathematical epiphanies oxford university press usa new ed edition december isbn boyer carl history of mathematics wiley edition march isbn concise history of mathematics from the concept of number to contemporary mathematics courant and robbins what is mathematics an elementary approach to ideas and methods oxford university press usa edition july isbn davis philip and hersh reuben the mathematical experience mariner books reprint edition january isbn gentle introduction to the world of mathematics eves howard an introduction to the history of mathematics sixth edition saunders isbn gullberg jan mathematics from the birth of numbers norton company st edition october isbn an encyclopedic overview of mathematics presented in clear simple language hazewinkel michiel ed 数学百科全书 kluwer academic publishers translated and expanded version of soviet mathematics encyclopedia in ten expensive volumes the most complete and authoritative work available also in paperback and on cd rom and online jourdain philip the nature of mathematics in the world of mathematics james newman editor dover isbn kline morris mathematical thought from ancient to modern times oxford university press usa paperback edition march isbn 牛津英语词典 second edition ed john simpson and edmund weiner clarendon press isbn the oxford dictionary of english etymology reprint isbn pappas theoni the joy of mathematics wide world publishing revised edition june isbn peterson ivars mathematical tourist new and updated snapshots of modern mathematics owl books isbn 参考网址 rusin dave the mathematical atlas 英文版 现代数学漫游 weisstein eric world of mathematics 一个在线的数学百科全书 planet math 另一个在线的数学百科全书 使用gfdl 允许和维基百科交换条目 mathforge 一个包含数学 物理 epistemath 数学知识 香港科技大学 数学网 一个以数学史为主的网站 怎样研习纯数学 或统计学 本科与基础研究课程参考书目 数学文化 主要面向大学生 大学老师和研究生 以及中学老师和学生 数学学习资源 互联网上数学学习资源和教学视频 英汉对照数学用语 archive 英汉对照数学用语 albany bureau of bilingual education see profile at archive
'''