---
layout:     post
title:      House Prices Advanced Regression Techniques
subtitle:   房价预测
date:       2017-12-06
author:     Tracy
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - PYTHON
    - 数据挖掘
---


- 本文源自kaggle中房价预测的一个非常高票的策略，原文作者是Pedro Marceino
- 原文地址:https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/notebook


- <font color ='orange'>本文的命题是通过79个解释性变量预测每个住宅的最终价格</font>

---

- 生活中最困难的事情就是了解自己
                        —————— 泰勒斯
                        
                        
- 而我想说的是理解你的数据是最困难的事情，它是非常乏味的。因此很多人会很容易就去忽略掉这个过程，直接下水。

- 那么我将要在下水之前先学习如何游泳，我尽我所能的去做一个全面而不详尽的数据分析。

- 我们所做的步骤是这样的：
    - 弄明白问题：我们将查看每一个变量并且对这个问题的意义和重要性进行哲学分析。
    - 单变量学习：我们只关注依赖变量(“SalePrice”)，并试着多了解一点。
    - 多变量研究：我们将尝试理解因变量和自变量之间的关系。
    - 基本的清洁：我们将清理数据集并处理丢失的数据、异常值和分类变量。
    - 检验假设：检查数据是否和多元分析方法的假设达到一致.
    
下面让我们开始进入吧


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings 
warnings.filterwarnings('ignore')

%matplotlib inline
```


```python
df_train = pd.read_csv('train.csv')
df_train.columns
```




    Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition', 'SalePrice'],
          dtype='object')



### 1 What can we expert?

为了理解我们的数据，我们需要观察每个变量并且弄明白他们对这个问题的意义和关联性。这个会非常耗时，但是非常有必要。

为了在我们分析中有些规范，我们用一下的列创建一个电子表格。

    - Variable - 变量名称
    - Type - 变量类型的识别，有两种变量类型：数值型和类别型。数值型的话意味着值是一些数字。类别型意味着值是一些分类。
    - Segment（划分） - ，我们定义三中划分：building、space、location。building意味着建筑物理特征，space意味房屋的内部空间，location就是物理位置不多说了。
    - Expectation（期望） - 我们希望该变量对房价的影响程度。我们使用类别标签‘高’，‘中’，‘低’作为可能值。
    - Conclusion （结论）-我们得出的该变量的重要性的结论。在大概浏览数据之后，我们认为这一栏和 “期望” 的值基本一致。
    - Comments - 我们看到的所有一般性评论
     
我们首先阅读了每一个变量的描述文件，同时思考这三个问题：
    - 我们买房子的时候会考虑这个因素吗？
    - 如果考虑的话，这个因素的重要程度如何？
    - 这个因素带来的信息在其他因素中出现过吗？
    
我们根据以上内容填好了电子表格，并且仔细观察了 “高期望” 的变量。然后绘制了这些变量和房价之间的散点图，填在了 “结论” 那一栏，也正巧就是对我们的期望值的校正。

我们总结出了四个对该问题起到至关重要的作用的变量：
    - OverallQual
    - YearBuilt
    - TotalBsmtSF
    - GrLivArea

我最终得到两个关于"building"变量（'OverallQual'、'YearBuilt'）,以及两个关于'space'变量（'TotalBsmtSF'、'GrLivArea'）。

### 现在最重要的事情是分析‘SalePrice’

就像我们去参加一个舞会，需要一个理由，通常女人酒等都是参加舞会的理由。现在'SalePrice'就像一个女人，好吧，我们暂且这样认定。我们要问'SalePrice'，“你能给我一些关于你的数据吗”，我需要依据这些来计算两个人的关系。


```python
df_train['SalePrice'].describe()
```




    count      1460.000000
    mean     180921.195890
    std       79442.502883
    min       34900.000000
    25%      129975.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64



看起来非常好！好像最低的价格也远远大于零。


```python
sns.distplot(df_train['SalePrice'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19a1b2ee8d0>



![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_9_1.png)


从直方图中可以看到：
    - 偏离正态分布
    - 数据正偏
    - 有峰值

数据偏度和峰度度量：


```python
print('Skewness:%f'% df_train['SalePrice'].skew())
print('Kurtosis:%f'% df_train['SalePrice'].kurt())
```

    Skewness:1.882876
    Kurtosis:6.536282
    

### 下面来找一些与数字型变量的关系，
Grlivarea 与 SalePrice 散点图


```python
data = df_train[['GrLivArea','SalePrice']]
data.plot.scatter(x='GrLivArea',y='SalePrice',ylim =(0,800000))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19a1ae3af28>




![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_13_1.png)


可以看出 SalePrice 和 GrLivArea 关系很密切，并且基本呈线性关系

那么我们看下'TotalBsmtSF'怎么样？


```python
data = df_train[['TotalBsmtSF','SalePrice']]
data.plot.scatter(x='TotalBsmtSF',y='SalePrice',ylim =(0,800000))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19a1afd29e8>




![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_15_1.png)


TotalBsmtSF  和 SalePrice 关系也很密切，从图中可以看出基本呈指数分布，但从最左侧的点可以看出特定情况下 TotalBsmtSF 对 SalePrice 没有产生影响

### 与类别型变量的关系
OverallQual’与‘SalePrice’箱型图


```python
data = df_train[['OverallQual','SalePrice']]
f,ax = plt.subplots(figsize =(8,6))
fig = sns.boxplot(x='OverallQual',y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)
```




    (-0.5, 9.5, 0, 800000)




![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_18_1.png)


可以看出 SalePrice 与 OverallQual 分布趋势相同。

YearBuilt 与 SalePrice 箱型图


```python
data = df_train[['YearBuilt','SalePrice']]
f,ax = plt.subplots(figsize =(20,8))
fig = sns.boxplot(x='YearBuilt',y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)
plt.xticks(rotation=90)
```




    (array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
             13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
             26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
             39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
             52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
             65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
             78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
             91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111]),
     <a list of 112 Text xticklabel objects>)




![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_20_1.png)


两个变量之间的关系没有很强的趋势性，但是可以看出建筑时间较短的房屋价格更高。

总结：

GrLivArea 和 TotalBsmtSF 与 SalePrice 似乎线性相关，并且都是正相关。 对于 TotalBsmtSF 线性关系的斜率十分的高。

OverallQual 和 YearBuilt 与 SalePrice 也有关系。OverallQual 的相关性更强, 箱型图显示了随着整体质量的增长，房价的增长趋势。

我们只分析了四个变量，但是还有许多其他变量我们也应该分析，这里的技巧在于选择正确的特征（特征选择）而不是定义他们之间的复杂关系（特征工程）。

### 保持冷静、客观分析
知道现在我们只是跟随自己的主观意识去判断一些变量，尽管我们努力为我们的分析提供客观的特性，但我们必须说，我们的出发点是主观的。

所以，让我们克服惰性，做一个更客观的分析

#### 相关系数矩阵


```python
corrmat = df_train.corr()
f,ax = plt.subplots(figsize =(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19a1cfd6e10>




![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_24_1.png)


首先两个红色的方块吸引到了我，第一个是 TotalBsmtSF 和 1stFlrSF 变量的相关系数，第二个是 GarageX 变量群。这两个示例都显示了这些变量之间很强的相关性。实际上，相关性的程度达到了一种多重共线性的情况。我们可以总结出这些变量几乎包含相同的信息，所以确实出现了多重共线性。

另一个引起注意的地方是 SalePrice 的相关性。我们可以看到我们之前分析的 GrLivArea，TotalBsmtSF和 OverallQual 的相关性很强，除此之外也有很多其他的变量应该进行考虑，这也是我们下一步的内容

#### SalePrice 相关系数矩阵


```python
# Get the rows of a DataFrame sorted by the `n` largest values of `columns
# 即用与saleprice相关性最高的10个参数作图
cols = corrmat.nlargest(10,'SalePrice')['SalePrice'].index
# np.corrcoef 返回皮尔逊相关性系数
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cbar=True,annot=True, square=True, fmt ='.2f', annot_kws={'size':10}, yticklabels= cols.values,xticklabels= cols.values)

```


![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_27_0.png)


OverallQual，GrLivArea 以及 TotalBsmtSF  与 SalePrice 有很强的相关性。

GarageCars 和 GarageArea 也是相关性比较强的变量. 车库中存储的车的数量是由车库的面积决定的，它们就像双胞胎，所以不需要专门区分 GarageCars 和 GarageAre，所以我们只需要其中的一个变量。这里我们选择了 GarageCars，因为它与 SalePrice 的相关性更高一些。

TotalBsmtSF  和 1stFloor 与上述情况相同，我们选择 TotalBsmtS 。

FullBath 几乎不需要考虑。

TotRmsAbvGrd 和 GrLivArea 也是变量中的双胞胎。

YearBuilt 和 SalePrice 相关性似乎不强。

#### SalePrice 和相关变量之间的散点图
有太多的信息在下面的散点图中了


```python
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols],size =2.5)
plt.show()
```


![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_30_0.png)


尽管我们已经知道了一些主要特征，这一丰富的散点图给了我们一个关于变量关系的合理想法。

其中，TotalBsmtSF 和 GrLiveArea 之间的散点图是很有意思的。我们可以看出这幅图中，一些点组成了线，就像边界一样。大部分点都分布在那条线下面，这也是可以解释的。地下室面积和地上居住面积可以相等，但是一般情况下不会希望有一个比地上居住面积还大的地下室。

SalePrice 和 YearBuilt 之间的散点图也值得我们思考。在 “点云” 的底部，我们可以观察到一个几乎呈指数函数的分布。我们也可以看到 “点云” 的上端也基本呈同样的分布趋势。并且可以注意到，近几年的点有超过这个上端的趋势。

### 缺失数据

#### 关于缺失数据需要思考的重要问题：
- 这一缺失数据的普遍性如何？
- 缺失数据是随机的还是有律可循？

这些问题的答案是很重要的，因为缺失数据意味着样本大小的缩减，这会阻止我们的分析进程。除此之外，以实质性的角度来说，我们需要保证对缺失数据的处理不会出现偏离或隐藏任何难以忽视的真相。


```python
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(20)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1453</td>
      <td>0.995205</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1406</td>
      <td>0.963014</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1369</td>
      <td>0.937671</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1179</td>
      <td>0.807534</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>690</td>
      <td>0.472603</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>259</td>
      <td>0.177397</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
      <td>0.000685</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



当超过 15% 的数据都缺失的时候，我们应该删掉相关变量且假设该变量并不存在。

根据这一条，一系列变量都应该删掉，例如 PoolQC，MiscFeature，Alley 等等，这些变量都不是很重要，因为他们基本都不是我们买房子时会考虑的因素。

GarageX 变量群的缺失数据量都相同，由于关于车库的最重要的信息都可以由 GarageCars 表达，并且这些数据只占缺失数据的 5%，我们也会删除上述的 GarageX 变量群。同样的逻辑也适用于 BsmtX 变量群。

对于 MasVnrArea 和 MasVnrType，我们可以认为这些因素并不重要。除此之外，他们和 YearBuilt 以及 OverallQual 都有很强的关联性，而这两个变量我们已经考虑过了。所以删除 MasVnrArea 和 MasVnrType 并不会丢失信息。

最后，由于 Electrical 中只有一个损失的观察值，所以我们删除这个观察值，但是保留这一变量。


```python
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,axis=1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()
```




    0



### 异常值

#### 单因素分析

这里的关键在于如何建立阈值，定义一个观察值为异常值。我们对数据进行正态化，意味着把数据值转换成均值为 0，方差为 1 的数据。


```python
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
```

    outer range (low) of the distribution:
    [[-1.83820775]
     [-1.83303414]
     [-1.80044422]
     [-1.78282123]
     [-1.77400974]
     [-1.62295562]
     [-1.6166617 ]
     [-1.58519209]
     [-1.58519209]
     [-1.57269236]]
    
    outer range (high) of the distribution:
    [[ 3.82758058]
     [ 4.0395221 ]
     [ 4.49473628]
     [ 4.70872962]
     [ 4.728631  ]
     [ 5.06034585]
     [ 5.42191907]
     [ 5.58987866]
     [ 7.10041987]
     [ 7.22629831]]
    

#### 进行正态化后，可以看出：

低范围的值都比较相似并且在 0 附近分布。

高范围的值离 0 很远，并且七点几的值远在正常范围之外。

### 双变量分析
GrLivArea 和 SalePrice 双变量分析


```python
data = df_train[['SalePrice','GrLivArea']]
data.plot.scatter(x='GrLivArea',y= 'SalePrice',ylim =(0,800000))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19a20f1fbe0>




![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_40_1.png)


#### 从图中可以看出：

有两个离群的 GrLivArea 值很高的数据，我们可以推测出现这种情况的原因。或许他们代表了农业地区，也就解释了低价。 这两个点很明显不能代表典型样例，所以我们将它们定义为异常值并删除。

图中顶部的两个点是七点几的观测值，他们虽然看起来像特殊情况，但是他们依然符合整体趋势，所以我们将其保留下来。


```python
#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
```


```python
data = df_train[['SalePrice','TotalBsmtSF']]
data.plot.scatter(x='TotalBsmtSF',y= 'SalePrice',ylim =(0,800000))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19a1f1828d0>




![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_43_1.png)


### 核心部分

“房价” 到底是谁？

这个问题的答案，需要我们验证根据数据基础进行多元分析的假设。

我们已经进行了数据清洗，并且发现了 SalePrice 的很多信息，现在我们要更进一步理解 SalePrice 如何遵循统计假设，可以让我们应用多元技术。

**应该测量 4 个假设量：**

    正态性

    同方差性

    线性

    相关错误缺失

**正态性：**

应主要关注以下两点：

    直方图 – 峰度和偏度。

    正态概率图 – 数据分布应紧密跟随代表正态分布的对角线。

### 绘制直方图和正态概率图：


```python
sns.distplot(df_train['SalePrice'],fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot=plt)
```


![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_47_0.png)



![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_47_1.png)


可以看出，房价分布不是正态的，显示了峰值，正偏度，但是并不跟随对角线。

可以用对数变换来解决这个问题

进行对数变换：


```python
df_train['SalePrice'] = np.log(df_train['SalePrice'])
```


```python
sns.distplot(df_train['SalePrice'],fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot=plt)
```


![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_50_0.png)



![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_50_1.png)


搞定了 

下面用GrLivArea绘制直方图和正态概率曲线图：


```python
df_train['GrLivArea']= np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'],fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'],plot=plt)
```


![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_52_0.png)



![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_52_1.png)



```python
sns.distplot(df_train['TotalBsmtSF'],fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'],plot=plt)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-53-d04b49a0c9e5> in <module>()
    ----> 1 sns.distplot(df_train['TotalBsmtSF'],fit=norm)
          2 fig = plt.figure()
          3 res = stats.probplot(df_train['TotalBsmtSF'],plot=plt)
    

    D:\Anaconda3\lib\site-packages\seaborn\distributions.py in distplot(a, bins, hist, kde, rug, fit, hist_kws, kde_kws, rug_kws, fit_kws, color, vertical, norm_hist, axlabel, label, ax)
        207     if hist:
        208         if bins is None:
    --> 209             bins = min(_freedman_diaconis_bins(a), 50)
        210         hist_kws.setdefault("alpha", 0.4)
        211         hist_kws.setdefault("normed", norm_hist)
    

    D:\Anaconda3\lib\site-packages\seaborn\distributions.py in _freedman_diaconis_bins(a)
         33         return int(np.sqrt(a.size))
         34     else:
    ---> 35         return int(np.ceil((a.max() - a.min()) / h))
         36 
         37 
    

    ValueError: cannot convert float NaN to integer



![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_53_1.png)


从图中可以看出：

显示出了偏度

大量为 0 的观察值（没有地下室的房屋）

含 0 的数据无法进行对数变换

我们建立了一个变量，可以得到有没有地下室的影响值（二值变量），我们选择忽略零值，只对非零值进行对数变换。这样我们既可以变换数据，也不会损失有没有地下室的影响。


```python
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
```

进行对数变换：


```python
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
```

绘制变换后的直方图和正态概率图：


```python
sns.distplot(df_train['TotalBsmtSF'],fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'],plot=plt)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-66-d04b49a0c9e5> in <module>()
    ----> 1 sns.distplot(df_train['TotalBsmtSF'],fit=norm)
          2 fig = plt.figure()
          3 res = stats.probplot(df_train['TotalBsmtSF'],plot=plt)
    

    D:\Anaconda3\lib\site-packages\seaborn\distributions.py in distplot(a, bins, hist, kde, rug, fit, hist_kws, kde_kws, rug_kws, fit_kws, color, vertical, norm_hist, axlabel, label, ax)
        207     if hist:
        208         if bins is None:
    --> 209             bins = min(_freedman_diaconis_bins(a), 50)
        210         hist_kws.setdefault("alpha", 0.4)
        211         hist_kws.setdefault("normed", norm_hist)
    

    D:\Anaconda3\lib\site-packages\seaborn\distributions.py in _freedman_diaconis_bins(a)
         33         return int(np.sqrt(a.size))
         34     else:
    ---> 35         return int(np.ceil((a.max() - a.min()) / h))
         36 
         37 
    

    ValueError: cannot convert float NaN to integer



![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_59_1.png)


同方差性：

最好的测量两个变量的同方差性的方法就是图像。
1.  SalePrice 和 GrLivArea 同方差性
绘制散点图：


```python
plt.scatter(df_train['GrLivArea'],df_train['SalePrice'])
```




    <matplotlib.collections.PathCollection at 0x19a1eff0c18>




![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_61_1.png)


2. SalePrice with TotalBsmtSF 同方差性

绘制散点图：


```python
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
```




    <matplotlib.collections.PathCollection at 0x19a2136ca90>




![png](http://op1xwtboy.bkt.clouddn.com/House%20Prices/output_63_1.png)


<img src ='https://static.leiphone.com/uploads/new/article/740_740/201704/58f7272e86be8.png?imageMogr2/format/jpg/quality/90'>

可以看出 SalePrice 在整个 TotalBsmtSF 变量范围内显示出了同等级别的变化.

将类别变量转换为虚拟变量：


```python
df_train = pd.get_dummies(df_train)
```

### 结论

整个方案中，我们使用了很多《多元数据分析》中提出的方法。我们对变量进行了哲学分析，不仅对 SalePrice 进行了单独分析，还结合了相关程度最高的变量进行分析。我们处理了缺失数据和异常值，我们验证了一些基础统计假设，并且将类别变量转换为虚拟变量。
但问题还没有结束，我们还需要预测房价的变化趋势，房价预测是否适合线性回归正则化的方法？是否适合组合方法？或者一些其他的方法？

希望你可以进行自己的探索发现。
