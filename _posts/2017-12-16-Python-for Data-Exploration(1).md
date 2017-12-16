---
layout:     post
title:      Python for Data  Exploration(1)
subtitle:   数据探索
date:       2017-12-06
author:     z-Tracy
header-img: img/post-bg-ioses.jpg
catalog: true
tags:
    - PYTHON
    - 数据挖掘
    - 数据探索
    - Pandas
---


- 这篇文章来学习下数据探索主要步骤，一般在我们搜索数据后进行数据挖掘前都需要考虑这样一些问题：数据集的数量和质量是否满足模型构建的要求？其中有什么明显的规律和趋势吗？
各个因素之间有什么样的关联性？
- 然后通过检验数据质量、绘制相应图标、计算某些特征量等手段对数据的结构和规律进行数据探索。
- 参考 python数据分析与挖掘实战

### 数据质量分析
- **首先应该就是检查数据中的脏数据，脏数据主要包含：缺失值、异常值、重复数据等。**

#### 缺失值分析
- 使用简单的统计分析，可以得到含有缺失值的属性个数，缺失率。对于缺失值处理有删除缺失值、对可能值进行插补、不处理三种情况

我最常用的方式是使用data.describe()查看情况，然后len(data)得出总的数据记录数。对比下缺了多少条数据。

#### 异常值分析
- 对变量进行描述性统计，最常用的是看最大值和最小值；也可以查看是否服从正态分布，如果服从的话一般异常值被定义为与平均值超过3倍标准差的值；另外一种也是我最常用的的方法，使用箱形图来分析，查看离群值。

#### 案例
- 我知道文字描述比较抽象的，我们使用真的数据来试一下看看！


```python
# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False
%matplotlib inline

catering_sale = 'catering_sale.xls' # 餐饮数据
data = pd.read_excel(catering_sale,index_col=u'日期')
len(data) - data.describe()[:1]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>销量</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



- 这边就显示了缺失值的数据为1个


```python
# 其实这样也可以，而且比较常用
data.isnull().sum()
```




    销量    1
    dtype: int64




```python
# 处理一下,用均值插入
data.fillna(value=data.mean(),inplace=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>销量</th>
    </tr>
    <tr>
      <th>日期</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-03-01</th>
      <td>51.0000</td>
    </tr>
    <tr>
      <th>2015-02-28</th>
      <td>2618.2000</td>
    </tr>
    <tr>
      <th>2015-02-27</th>
      <td>2608.4000</td>
    </tr>
    <tr>
      <th>2015-02-26</th>
      <td>2651.9000</td>
    </tr>
    <tr>
      <th>2015-02-25</th>
      <td>3442.1000</td>
    </tr>
    <tr>
      <th>2015-02-24</th>
      <td>3393.1000</td>
    </tr>
    <tr>
      <th>2015-02-23</th>
      <td>3136.6000</td>
    </tr>
    <tr>
      <th>2015-02-22</th>
      <td>3744.1000</td>
    </tr>
    <tr>
      <th>2015-02-21</th>
      <td>6607.4000</td>
    </tr>
    <tr>
      <th>2015-02-20</th>
      <td>4060.3000</td>
    </tr>
    <tr>
      <th>2015-02-19</th>
      <td>3614.7000</td>
    </tr>
    <tr>
      <th>2015-02-18</th>
      <td>3295.5000</td>
    </tr>
    <tr>
      <th>2015-02-16</th>
      <td>2332.1000</td>
    </tr>
    <tr>
      <th>2015-02-15</th>
      <td>2699.3000</td>
    </tr>
    <tr>
      <th>2015-02-14</th>
      <td>2755.2147</td>
    </tr>
    <tr>
      <th>2015-02-13</th>
      <td>3036.8000</td>
    </tr>
    <tr>
      <th>2015-02-12</th>
      <td>865.0000</td>
    </tr>
    <tr>
      <th>2015-02-11</th>
      <td>3014.3000</td>
    </tr>
    <tr>
      <th>2015-02-10</th>
      <td>2742.8000</td>
    </tr>
    <tr>
      <th>2015-02-09</th>
      <td>2173.5000</td>
    </tr>
    <tr>
      <th>2015-02-08</th>
      <td>3161.8000</td>
    </tr>
    <tr>
      <th>2015-02-07</th>
      <td>3023.8000</td>
    </tr>
    <tr>
      <th>2015-02-06</th>
      <td>2998.1000</td>
    </tr>
    <tr>
      <th>2015-02-05</th>
      <td>2805.9000</td>
    </tr>
    <tr>
      <th>2015-02-04</th>
      <td>2383.4000</td>
    </tr>
    <tr>
      <th>2015-02-03</th>
      <td>2620.2000</td>
    </tr>
    <tr>
      <th>2015-02-02</th>
      <td>2600.0000</td>
    </tr>
    <tr>
      <th>2015-02-01</th>
      <td>2358.6000</td>
    </tr>
    <tr>
      <th>2015-01-31</th>
      <td>2682.2000</td>
    </tr>
    <tr>
      <th>2015-01-30</th>
      <td>2766.8000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2014-08-31</th>
      <td>3494.7000</td>
    </tr>
    <tr>
      <th>2014-08-30</th>
      <td>3691.9000</td>
    </tr>
    <tr>
      <th>2014-08-29</th>
      <td>2929.5000</td>
    </tr>
    <tr>
      <th>2014-08-28</th>
      <td>2760.6000</td>
    </tr>
    <tr>
      <th>2014-08-27</th>
      <td>2593.7000</td>
    </tr>
    <tr>
      <th>2014-08-26</th>
      <td>2884.4000</td>
    </tr>
    <tr>
      <th>2014-08-25</th>
      <td>2591.3000</td>
    </tr>
    <tr>
      <th>2014-08-24</th>
      <td>3022.6000</td>
    </tr>
    <tr>
      <th>2014-08-23</th>
      <td>3052.1000</td>
    </tr>
    <tr>
      <th>2014-08-22</th>
      <td>2789.2000</td>
    </tr>
    <tr>
      <th>2014-08-21</th>
      <td>2909.8000</td>
    </tr>
    <tr>
      <th>2014-08-20</th>
      <td>2326.8000</td>
    </tr>
    <tr>
      <th>2014-08-19</th>
      <td>2453.1000</td>
    </tr>
    <tr>
      <th>2014-08-18</th>
      <td>2351.2000</td>
    </tr>
    <tr>
      <th>2014-08-17</th>
      <td>3279.1000</td>
    </tr>
    <tr>
      <th>2014-08-16</th>
      <td>3381.9000</td>
    </tr>
    <tr>
      <th>2014-08-15</th>
      <td>2988.1000</td>
    </tr>
    <tr>
      <th>2014-08-14</th>
      <td>2577.7000</td>
    </tr>
    <tr>
      <th>2014-08-13</th>
      <td>2332.3000</td>
    </tr>
    <tr>
      <th>2014-08-12</th>
      <td>2518.6000</td>
    </tr>
    <tr>
      <th>2014-08-11</th>
      <td>2697.5000</td>
    </tr>
    <tr>
      <th>2014-08-10</th>
      <td>3244.7000</td>
    </tr>
    <tr>
      <th>2014-08-09</th>
      <td>3346.7000</td>
    </tr>
    <tr>
      <th>2014-08-08</th>
      <td>2900.6000</td>
    </tr>
    <tr>
      <th>2014-08-07</th>
      <td>2759.1000</td>
    </tr>
    <tr>
      <th>2014-08-06</th>
      <td>2915.8000</td>
    </tr>
    <tr>
      <th>2014-08-05</th>
      <td>2618.1000</td>
    </tr>
    <tr>
      <th>2014-08-04</th>
      <td>2993.0000</td>
    </tr>
    <tr>
      <th>2014-08-03</th>
      <td>3436.4000</td>
    </tr>
    <tr>
      <th>2014-08-02</th>
      <td>2261.7000</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 1 columns</p>
</div>




```python
plt.figure() #建立图像
p = data.boxplot(return_type='dict') #画箱线图，直接使用DataFrame的方法

# 这边的用法如果不清楚 可以直接查看下p即boxplot函数返回的值，就比较清楚了。
x = p['fliers'][0].get_xdata() # 'flies'即为异常值的标签，然后通过get_xdata()返回x坐标
y = p['fliers'][0].get_ydata()
y.sort() #从小到大排序，该方法直接改变原对象

#用annotate添加注释
#其中有些相近的点，注解会出现重叠，难以看清，需要一些技巧来控制。
#以下参数都是经过调试的，需要具体问题具体调试。
for i in range(len(x)): 
    if i>0:
        plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
    else:
        plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))

plt.show() #展示箱线图
```


![png](http://op1xwtboy.bkt.clouddn.com/data%20exploration/output_10_0.png)


- 通过箱形图异常值一目了然了吧，共有7个，当然你可以根据自己的业务比如将865、4060.3、4065.2归为正常值，其他几个定为异常值。
根据你的业务情况来。


```python
# 把几个异常值也处理一下
import numpy as np
data.sort_values(u'销量',ascending=False,inplace=True)
data[-3:] = data.mean()[0]
data[:2] = data.mean()[0]
```

顺便说一下，同样如果你比较喜欢用seaborn也可以用，不过异常值的标记自己琢磨一下，我知道你可以的


```python
import seaborn as sns
f,ax =plt.subplots()
t = sns.boxplot(data=data)
```


![png](http://op1xwtboy.bkt.clouddn.com/data%20exploration/output_14_0.png)


### 数据特征分析

- 用之前的办法基本可以处理了数据质量问题，下来要来聊聊数据特征分析。这个对于后续的数据分析和建模还是很有启发性的工作。

#### 分布分析

分布分析能揭示数据的分布特征和分布类型。对于定量数据，发现其分布形式是对称的还是非对称的，发现某些特大或特效的可疑值，主要还是通过绘制频率分布图
、直方图、茎叶图进行只管分析；对于定性的分类数据，可以用饼图和条形图直观显示分布情况。

1. 对于定量变量而言，选择组数和组宽是做频率分布分析最重要的问题。一般按照以下步骤。
    - 求极差
    - 决定组距和组数
    - 决定分点
    - 列出频率分布表
    - 绘制频率分布直方图

- 我们还是用之前的数据来实现一下看看好吧。


```python
# 极差
Range = data.max()-data.min()
print('最大值是：%.2f' %(data.max()))
print('最小值是：%.2f' %(data.min()))
print('极差为：{:0.2f}'.format(Range.values[0]))
```

    最大值是：4065.20
    最小值是：865.00
    极差为：3200.20
    


```python
# 组数，可根据业务来确定组距，这里设为500
# 那么组数就可以计算为：
k = Range/500
print('组数为：{:0.0f}组'.format(k.values[0]))
```

    组数为：6组
    

- 绘制频率直方图


```python
# f = plt.subplots(figsize=(5,4))
f = plt.hist(data['销量'],bins=7,normed=True,facecolor='b')
plt.xlabel(r'日销售额/元')
plt.ylabel(r'频率')
plt.title(r'销售额分布直方图')
```




    <matplotlib.text.Text at 0x1e4784359e8>




![png](http://op1xwtboy.bkt.clouddn.com/data%20exploration/output_24_1.png)


#### 定性数据的分布分析

- 对于定性变量的分布分析相对比较简单，根据变量的分类类型就可以尝试用饼图、条形图等一般图形来观察。这里也不多做解释。

#### 对比分析

- 对比分析主要有两种形式
    - **绝对数比较**：利用绝对数进行对比寻找差异的方法
    - **相对数比较**：由两个有联系的指标进行对比计算，用来反映客观现象之间数量联系程度的综合指标。

#### 相关性分析

- 分析连续变量之间的线性相关程度的强弱，用适当的统计指标表示出来的过程称为相关分析。
    1. 直接绘制散点图观察(plot.scatter)
    2. 绘制散点图举证(sns.pairplot)
    3. 计算相关系数，比较常用的有Pearson相关系数、Spearman秩相关系数和判定系数。
      （其实Pearson相关系数要求连续变量的取值服从正态分布，若不服从则可采用Speraman相关系数）

- 在python直接有对应的函数，下面我们用实例来看下。


```python
from __future__ import print_function
catering_sale = 'catering_sale_all.xls'
data = pd.read_excel(catering_sale,index_col=u'日期')

data.corr(method='pearson')['百合酱蒸凤爪'].sort_values(ascending = False)
# data['百合酱蒸凤爪'].corr(data['翡翠蒸香茜饺'])
```




    百合酱蒸凤爪     1.000000
    乐膳真味鸡      0.455638
    原汁原味菜心     0.428316
    生炒菜心       0.308496
    铁板酸菜豆腐     0.204898
    香煎韭菜饺      0.127448
    蜜汁焗餐包      0.098085
    金银蒜汁蒸排骨    0.016799
    翡翠蒸香茜饺     0.009206
    香煎罗卜糕     -0.090276
    Name: 百合酱蒸凤爪, dtype: float64



这里看到，如果一个用户点了'百合酱蒸凤爪',则点其他菜品的概率排序是这样的，'乐膳真味鸡'>'原汁原味菜心'>'生炒菜心'>'铁板酸菜豆腐'……

- 用图表更加直观


```python
data_corr = data.corr()
f,ax = plt.subplots(figsize =(6,6))

sns.heatmap(data_corr,square=True,fmt ='.2f',annot=True,annot_kws={'size':10})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b7c9ad35c0>




![png](http://op1xwtboy.bkt.clouddn.com/data%20exploration/output_34_1.png)


很明显相关性程度很低

### python 主要数据探索函数

pandas提供大量的数据探索相关的函数。

#### 基本统计特征函数(Pandas)

>- 计算数据总和:data.sum()
- 计算数据的均数:data.mean()
- 计算数据的方差：data.var()
- 计算数据的标准差：data.std()
- 计算数据的相关系数矩阵：data.corr()
- 计算数据的协方差：data.cov()
- 计算数据的偏度和峰度：data.skew()/data.kurt()

#### 拓展统计特征函数(Pandas)

> - 依次给出前1、2、3、……、n个数的和 ：data.cumsum()
- 依次给出前1、2、3、……、n个数的积 ：data.cumprod()
- 依次给出前1、2、3、……、n个数的最大值 ：data.cummax()
- 依次给出前1、2、3、……、n个数的最小值 ：data.cummin()
- 滚动计算数据的总和（按列计算）: pd.rolling_sum(data,k)（k代表指定的烈属）
- 滚动计算数据的均值（按列计算）: pd.rolling_mean(data,k)（k代表指定的烈属）

另外其他还是的rolling_ 系列函数分别是:rolling_var(),rolling_std(),
    rolling_corr(),rolling_cov(),rolling_skew(),rolling_kurt()

#### 统计作图函数

- 主要的库有matplotlib和seaborn库，其中seaborn也是基于matplotlib创建的。另外交互式图表可以尝试使用pycharts这个库，基于百度的echart。
- 主要的方式是直接调用matplotlib，例如plt.boxplot();还有另一个方式是用Dataframe内置方法，例如data.plot(kind="box")
- 具体的方式也比较简单，使用时可以直接查询相应的官方文档。

### 总结

- 这一篇主要理了下数据分析前的工作——数据探索的内容，以及相应的pandas对应的用法和作图方法。从而能够发现数据中的一些规律和趋势，为后续建模找到方向。另外数据分析前很重要的一项工作——数据清理工作。对于缺失值、异常值做一些处理。如果这篇文章对你有用或者还有补充希望能留言，让我和其他读者能够学习到。
