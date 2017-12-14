---
layout:     post
title:      python文本情感分析
subtitle:   电商产品评论数据情感分析
date:       2017-12-14
author:     Tracy
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - PYTHON
    - 数据挖掘
	- 文本分析
---
- 参考文档:http://www.tipdm.org/u/cms/www/201511/23155400syj0.pdf
```python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
```
#### 评论预处理-文本去重
  1. 剔除大量无价值的词
  2. 文本去重，去除平台默认评价的词；同一用户重复评价了多条相同评价；


```python
def drop_duplicates():
    data = pd.read_table('meidi_jd.txt',encoding='utf-8',header=None)
    
    l1 = len(data)
    data = pd.DataFrame(data[0].unique())
    l2 = len(data)
    data.to_csv('meidi_jd_process_1.txt',index=False,header=False)
    print('删除了{0}条重复数据'.format(l1-l2),'\n剩下{0}条评论'.format(l2))
    
drop_duplicates()
```

    删除了2725条重复数据 
    剩下53050条评论
    


```python
data = pd.read_table('meidi_jd_process_1.txt',encoding='ANSI')
type(data['x'])
```




    pandas.core.series.Series



#### 机械压缩去词
    1. 机械压缩去词世界要处理的语料是预料中有连续累赘重复的部分，例如“为什么为什么为什么安装费这么贵，毫无道理”这里面的“为什么”重复词，并且只对头尾的连续重复词去除。中间的重复词则不做处理，因为中间一般非常少见，例如“安装师傅滔滔不绝的向我讲解这款热水器的功能”，这句话如果中间做重复去除，将会去除“滔”字，明显不符合。


```python

```

#### 短句删除


```python

```

#### 停用词过滤


```python

```

#### 情感倾向性分析
- 评定一个句子情感为积极的还是消极的
- 有snownlp库、波森中文语义 api ：bosonnlp.com 、腾讯文智接口：nlp.qq.com 等
- snownlp 参考：http://blog.sciencenet.cn/blog-377709-1062960.html


```python
%%time
# 利用snownlp计算情感值，运行时间需要6分钟
from snownlp import SnowNLP
def isPosOrNev(sentences):
    s = SnowNLP(sentences).sentiments   
    return s
# data['scores'] = data.x.apply(lambda x : SnowNLP(str(x)).sentiments)
data['scores'] = data.x.apply(isPosOrNev)
data
```

    Wall time: 6min 26s
    

- 分类导出积极与消极的语料


```python
data[data['scores']>.6]['x'].to_csv('meidi_jd_pos.txt',index=False,header=False,encoding='utf-8')
data[data['scores']<.4]['x'].to_csv('meidi_jd_neg.txt',index=False,header=False,encoding='utf-8')
! type meidi_jd_pos.txt
```

#### 使用PYTHON的Gensim完成LDA分析


```python
%%time
negfile = 'meidi_jd_neg_cut.txt'
posfile = 'meidi_jd_pos_cut.txt'
# 由于停用词还未完成，这里先注释
stopfile = 'stoplist.txt'

neg = pd.read_csv(negfile,encoding='utf-8',header=None)
pos = pd.read_csv(posfile,encoding='utf-8',header=None)
stop = pd.read_csv(stopfile,encoding='utf-8',header=None,sep=r'\n')
stop = [' ','']+list(stop[0])


neg[1] = neg[0].apply(lambda s : s.split(' '))
neg[2] = neg[1].apply(lambda x:[i for i in x if i not in stop]) # 逐词判断是否是停用词 
pos[1] = pos[0].apply(lambda s : s.split(' '))
pos[2] = pos[1].apply(lambda x:[i for i in x if i not in stop]) # 逐词判断是否是停用词 
pos[2]
```

    D:\Anaconda3\lib\site-packages\ipykernel\__main__.py:8: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
    

    Wall time: 25.1 s
    


```python
from gensim import corpora,models

# 负面主题分析
neg_dict = corpora.Dictionary(neg[2]) #建立词典
neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]] # 建立语料库 
neg_lda = models.LdaModel(neg_corpus,num_topics=5,id2word=neg_dict) # lAD模型训练
for i in range(5):
    print(neg_lda.print_topic(i)) # 输出每个主题
```

    D:\Anaconda3\lib\site-packages\gensim\utils.py:860: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    

    0.029*"安装" + 0.019*"安装费" + 0.017*"热水器" + 0.016*"美的" + 0.013*"遥控" + 0.012*"元" + 0.011*"贵" + 0.011*"东西" + 0.011*"装" + 0.009*"不错"
    0.039*"安装" + 0.018*"送货" + 0.016*"加热" + 0.014*"评价" + 0.014*"收到" + 0.011*"快递" + 0.011*"京东" + 0.011*"不错" + 0.011*"货" + 0.009*"慢"
    0.089*"安装" + 0.036*"师傅" + 0.028*"热水器" + 0.022*"美的" + 0.020*"不错" + 0.016*"售后" + 0.015*"京东" + 0.011*"服务" + 0.011*"送货" + 0.010*"收费"
    0.067*"安装" + 0.024*"元" + 0.014*"配件" + 0.012*"不错" + 0.010*"师傅" + 0.010*"水阀" + 0.009*"花" + 0.009*"费用" + 0.009*"花洒" + 0.008*"混"
    0.023*"安装" + 0.020*"加热" + 0.017*"度" + 0.016*"热水器" + 0.013*"号" + 0.011*"师傅" + 0.011*"小时" + 0.011*"温度" + 0.010*"热水" + 0.009*"第二天"
    


```python
data1[data1.str.contains('花洒')]
```




    11                      很 好 ， 就是 花洒 出水 太小 了 的 感觉 ， 自己 换 一个
    50       发 的 配件 没有 水管 呢 ， 很多 配件 要 自己 买 ， 而且 安装 师傅 打洞 啊 ...
    54                               加热 挺快 ， 用 起来 满意 ， 花洒 比较 次
    64       东西 不错 ， 就是 安装 时 也 没 混 水阀 ， 只有 一个 很 不好 的 花洒 ， 还...
    99       最 满意 的 一次 , 唯一 觉得 差一点 的话 , 就是 花洒 出水量 不 大 , 估计 ...
    103      买 了 两个 ， 但 安装费 真的 太贵 了 ！ ！ 一台 要 300 多 ！ ！ 我 让 ...
    140      出水管 和 花洒 不好 用 ， 水 很小 ， 都 洗 发烧 了 。 自己 换 了 一套 花洒...
    157                                        花洒 不 怎样 ， 其它 OK
    165      不错 吧 ， 看 了 其他人 的 评价 都 不敢 叫 师傅 上门 安装 ， 自己 搞掂 . ...
    190      装 上 当天 就 开始 使用 了 ， 升温 也 挺快 的 ， 热水 温度 挺 好 ， 花洒 ...
    239      热水器 非常 漂亮 ， 也 非常 好用 ！ 出水 快 ， 而且 烧水 也 快 ， 还 非常 ...
    251      商品 不错 ， 就是 花洒 太差 了 ， 出水 很小 啊 ， 根本 没有 办法 用 ， 我 ...
    269                        真心 给力 ！ 就是 有点 费电 ！ 花洒 不是 很 理想 ！
    300      东西 很 好 ， 用 起来 非常 爽 ， 第二天 就 来 评价 。 买回来 自己 安装 的 ...
    304      热水器 价格 不算 贵 ， 免费 装 ， 但是 配件 基本上 要 300 + ， 自己 买 ...
    359                                    还 可以 ， 不过 花洒 孔 太小 了
    363      装上 后 使用 ， 还 可以 。 保温 也 不错 ， 就是 加热 时间 比较 长 ， 配 的...
    373                                   花洒 水流 小 了 点   其它 还 行
    459      样子 精巧 。 虽说 是 标配 花洒 等 ， 安装 的 时候 还是 收 了 180 。 而且...
    465      送货 速度 快 ， 送货员 、 安装工 态度 好 ， 但 安装费 高 。 厂商 提供 用料 ...
    489      热水器 今天 装 好 ， 东西 不错 ， 是 正品 ， 刚才 试 了 下 ， 热水 来 的 ...
    555      自己 借 了 个 电钻 买 了 钻头 和 水管 什么 的 自己 装好 了 ， 除了 花洒共花...
    662                   花洒 太小 了 ， 出水 太慢 ， 洗个 澡 要 老半天 ！ 其它 还好
    715          1 、 很 实用 ； 2 、 适合 两人 使用 ； 3 、 花洒 和 混水 器挺 粗糙 的
    716      目前 使用 发现 热 有点 慢 ， 但 相信 美的 的 质量 和 售后 态度 。 那个 附带...
    744      送 的 花洒 比较 一般 了 ， 不能 换挡 ， 热水器 还是 可以 的 ， 加热 速度 挺...
    760                                                花洒 比较简单
    822      首先 烧水 的 速度慢 ， 每次 洗 之前 基本上 要 烧 一两个 小时 - 它 才 会 自...
    849      用 着 还 可以 ， 也许 是 花洒 太大 了 ， 出水量 大 的 缘故 ， 洗 了 一会 ...
    899      快递 太给力 了   1 天 就 到 了   京东 快递 就是 给力 啊     师傅 很快...
                                   ...                        
    15557    包装 里面 没有 混 水阀 和 花洒 ， 都 是 在 安装 师傅 那买 的 ， 安装费 花 ...
    15559    先说 安装 是 免费 的 。 配件 却 要 几百块 ， 但 安装 人员 说 可以 自己 买 ...
    15560    先说 安装 是 免费 的 。 配件 却 要 几百块 ， 但 安装 人员 说 可以 自己 买 ...
    15576    目前 用 着 还 不错 ， 师傅 有 挺 认真 讲解 如何 设置 。 就是 觉得 自己 要 ...
    15580    新房 装修 还 没有 使用 ， 安装 也 挺快 ， 花 了 100 块钱 安装 花洒 的 费用 ！
    15594    总体 感觉 不错 吧 ， 送货 很快 ， 安装 也 比较 速度 。 。 。 但是 有 一点 ...
    15673    东西 很 不错 ， 电容 触控 面板 很 灵敏 ， 加热 速度 快 ， 保温 性能 也 不错...
    15674    东西 很 不错 ， 电容 触控 面板 很 灵敏 ， 遥控器 用处 不 大 。 加热 速度 快...
    15694    花洒 没配 ， 全套 安装 258 元 ， 略贵 ， 不过 师傅 晚上 八点 多 还 专门 ...
    15751    还 没用   , 就是 有点 大 . 长度 872 , 真是 一点 不差   , 还 有点 ...
    15811    安装 乱收费 ， 自己 买 材料 一样 要 安装费 。 什么 首次 免费 安装 都 是 空淡...
    15869    分 水龙头 ， 软管 和 花洒 没有 ， 自己 配 的 。 \ n 操作 起来 还 挺 简单...
    15872    买 了 没 几天 就 降价 * 元 ， 觉得 挺亏 的 。 没有 花洒 ， 还 自己 专门 ...
    15919                            可能 家里 水压 唔 够 ， 花洒 出水 比较 细
    16006    今天 师傅 安装 的 ， 几根 管子 要 350 ， 另外 50 是 安装 花洒 的 安装费...
    16017    选择 升数 的 时候 犹豫 了 很 久 ， 怕 自己 洗澡 费水 不够 用选 了 60 升 ...
    16031    优点 ： \ n 热水器 和 安装 都 没有 质量 问题 \ n 缺点 ： \ n 送 的 ...
    16100    安装 速度 很快 。 。 就是 花 洒水 太小 了 。 。 。 需要 换个 花洒 。 。 其...
    16185    选 的 星期六 来 送货     隔天 安装     安装 花 了 36     大部分 材...
    16377        我家 的 新 热水器 ， 我 很 喜欢 。 花洒 是 我 自己 的 ， 没用 赠送 的 。
    16398    今天 安装 师傅 过来 ， 打开 以后 ， 才 知道 里面 没有 混 水阀 ， 软管 和 花...
    16400                         连个 花洒 都 要 自己 买   这 都 什么 商家 啊
    16460    加热 速度 挺快   发过来 的 时候 漏 了 花洒 等   联系 客服 马上 补发 了  ...
    16493    这 款 热水器 16A 的 插头 要 注意 ， 如果 不是 墙上 有 大功率 空调 插座 ，...
    16512               安装费 有点 高 ， 自己 更换 了 花洒 再 加上 安装费 ， 安装费 *
    16587                                 出水 范围 小 ， 花洒 喷射 范围 小
    16591                         产品 不错 ， 就是 配套 的 花洒 质量 有些 差 了
    16606             质量 没 问题 的 ， 只是 送 的 花洒 不 杂样 啊 。 。 出水 太小 了
    16667    全部 自己 安装 ， 平安 PPR 管 焊接 ， 电锤 订好 距离 打孔 安装 热水器 ， ...
    16670    快递 员 很 好 ， 帮 我 把 热水器 搬 到 二楼 没有 怨言 ， 赞 一个 ！ 安装 ...
    Name: 0, dtype: object




```python
# 正面主题分析
pos_dict = corpora.Dictionary(pos[2])
pos_corpus = [pos_dict.doc2bow(i) for i in pos[2]]
pos_lda = models.LdaModel(pos_corpus,num_topics=3,id2word=pos_dict)
for i in range(3):
    print(pos_lda.print_topic(i))
```

    0.161*"不错" + 0.049*"东西" + 0.032*"挺" + 0.027*"价格" + 0.025*"实惠" + 0.025*"质量" + 0.025*"性价比" + 0.024*"热水器" + 0.023*"便宜" + 0.022*"高"
    0.051*"不错" + 0.040*"京东" + 0.031*"值得" + 0.026*"美的" + 0.021*"品牌" + 0.019*"实用" + 0.019*"购买" + 0.019*"信赖" + 0.014*"感觉" + 0.013*"热水器"
    0.083*"安装" + 0.055*"不错" + 0.037*"送货" + 0.032*"很快" + 0.030*"速度" + 0.019*"师傅" + 0.019*"服务" + 0.016*"外观" + 0.015*"加热" + 0.013*"效果"
    


```python

```