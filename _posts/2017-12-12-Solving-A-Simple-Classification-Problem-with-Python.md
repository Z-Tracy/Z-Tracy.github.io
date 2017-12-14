---
layout:     post
title:      Solving A Simple Classification Problem with Python Fruits Lovers’ Edition
subtitle:   得到
date:       2017-12-06
author:     Z-Tracy
header-img: img/post-bg-mma-2.jpg
catalog: true
tags:
    - PYTHON
    - 数据挖掘
---
本文为译文
- 原文链接：https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2
- 原作者：Susan Li  
- 主页地址：https://towardsdatascience.com/@actsusanli?source=post_header_lockup

- 本文我们将实现几个python中scikit-learn中的机器学习算法。使用一个简单的数据集作为分类器训练集去区分不同类型的水果。
- 这篇文章的目的是为了识别最适合手头问题的机器学习算法，从而我们想要去比较不同的算法，选择表现最好的一个。
- 开始吧!

- 数据来源爱丁堡大学的一位同学—— [Dr. Iain Murray](http://homepages.inf.ed.ac.uk/imurray2/)
他买了大量的水果进行测量得到这份数据
- 数据可以到本文作者的github中下载——[here](https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/fruit_data_with_colors.txt)

先预览下数据好了


```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fruits = pd.read_table('fruit_data_with_colors.txt')
fruits.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fruit_label</th>
      <th>fruit_name</th>
      <th>fruit_subtype</th>
      <th>mass</th>
      <th>width</th>
      <th>height</th>
      <th>color_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>apple</td>
      <td>granny_smith</td>
      <td>192</td>
      <td>8.4</td>
      <td>7.3</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>apple</td>
      <td>granny_smith</td>
      <td>180</td>
      <td>8.0</td>
      <td>6.8</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>apple</td>
      <td>granny_smith</td>
      <td>176</td>
      <td>7.4</td>
      <td>7.2</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>mandarin</td>
      <td>mandarin</td>
      <td>86</td>
      <td>6.2</td>
      <td>4.7</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>mandarin</td>
      <td>mandarin</td>
      <td>84</td>
      <td>6.0</td>
      <td>4.6</td>
      <td>0.79</td>
    </tr>
  </tbody>
</table>
</div>



每一行代表一个水果，表格列代表着特征值

总过有59个水果和7个特征值


```python
fruits.shape
```




    (59, 7)



- 查看下一个哪几种种水果？


```python
print(fruits.fruit_name.unique())
```

    ['apple' 'mandarin' 'orange' 'lemon']
    

- 各种类水果分别有多少个？


```python
fruits.groupby('fruit_name').size()
```




    fruit_name
    apple       19
    lemon       16
    mandarin     5
    orange      19
    dtype: int64




```python
import seaborn as sns
sns.countplot(fruits['fruit_name'],label= 'Count')
plt.show()
```


![png](http://op1xwtboy.bkt.clouddn.com/output_11_0.png)


- 线箱图将使我们更清楚的了解每个数字变量的分布情况


```python
fruits.drop('fruit_label',axis=1).plot(kind='box',subplots = True,
                                      layout=(2,2),sharex = False,
                                      sharey = False,figsize= (9,9),
                                      title ='Box Plot for each input variable')
plt.savefig('fruits_box')
plt.show()
```


![png](http://op1xwtboy.bkt.clouddn.com/output_13_0.png)


- 上图看起来貌似颜色这个特征有点接近正态分布，我们再看下直方图


```python
import pylab as pl
fruits.drop('fruit_label',axis=1).hist(bins=30,figsize=(9,9))
pl.suptitle('Histogram for each numeric input variable')
plt.savefig('fruits_hist')
plt.show()
```


![png](http://op1xwtboy.bkt.clouddn.com/output_15_0.png)


- 有一些属性对存在相关性（比如mass和width）


```python
# Draw a matrix of scatter plots
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
feature_name = ['mass','width','height','color_score']
X = fruits[feature_name]
y = fruits['fruit_label']

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X,c=y,marker='o',s=40,hist_kwds={'bins':15},
                            figsize=(9,9),cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
```




    <matplotlib.text.Text at 0x21ac4b3f358>




![png](http://op1xwtboy.bkt.clouddn.com/output_17_1.png)


- 上图暂时还没有理解作者想要表达什么


```python
fruits.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fruit_label</th>
      <th>mass</th>
      <th>width</th>
      <th>height</th>
      <th>color_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>59.000000</td>
      <td>59.000000</td>
      <td>59.000000</td>
      <td>59.000000</td>
      <td>59.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.542373</td>
      <td>163.118644</td>
      <td>7.105085</td>
      <td>7.693220</td>
      <td>0.762881</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.208048</td>
      <td>55.018832</td>
      <td>0.816938</td>
      <td>1.361017</td>
      <td>0.076857</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>76.000000</td>
      <td>5.800000</td>
      <td>4.000000</td>
      <td>0.550000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>140.000000</td>
      <td>6.600000</td>
      <td>7.200000</td>
      <td>0.720000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>158.000000</td>
      <td>7.200000</td>
      <td>7.600000</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>177.000000</td>
      <td>7.500000</td>
      <td>8.200000</td>
      <td>0.810000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>362.000000</td>
      <td>9.600000</td>
      <td>10.500000</td>
      <td>0.930000</td>
    </tr>
  </tbody>
</table>
</div>



- 我们看到这些数据属性变量不在相同的比例范围内，需要我们去缩放这些数据到我们训练集需要的比例

#### 创建训练集和测试集以及应用缩放比例


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

# 将属性缩放到一个指定的最大和最小值（通常是1-0）之间，这可以通过preprocessing.MinMaxScaler类实现。
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 构建模型
#### Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

print("Accuracy of Logistic regression classifier on training set:{0:.2f}"
      .format(logreg.score(X_train,y_train)))
print("Accuracy of Logistic regression classifier on test set:{0:.2f}"
      .format(logreg.score(X_test,y_test)))
```

    Accuracy of Logistic regression classifier on training set:0.70
    Accuracy of Logistic regression classifier on training set:0.40
    

#### Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train,y_train)

print("Accuracy of Decision Tree classifier on training set:{0:.2f}"
      .format(clf.score(X_train,y_train)))
print("Accuracy of Decision Tree classifier on test set:{0:.2f}"
      .format(clf.score(X_test,y_test)))
```

    Accuracy of Decision Tree classifier on training set:1.00
    Accuracy of Decision Tree classifier on training set:0.87
    

#### K-Nearest Neighbors


```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print("Accuracy of K-NN classifier on training set:{0:.2f}"
      .format(knn.score(X_train,y_train)))
print("Accuracy of K-NN classifier on test set:{0:.2f}"
      .format(knn.score(X_test,y_test)))
```

    Accuracy of K-NN classifier on training set:0.95
    Accuracy of K-NN classifier on training set:1.00
    

#### Linear Discriminant Analysis


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
print("Accuracy of LDA classifier on training set:{0:.2f}"
      .format(lda.score(X_train,y_train)))
print("Accuracy of LDA classifier on test set:{0:.2f}"
      .format(lda.score(X_test,y_test)))
```

    Accuracy of LDA classifier on training set:0.86
    Accuracy of LDA classifier on training set:0.67
    


```python
#### Gaussian Naive Bayes(高斯朴素贝叶斯)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)
print("Accuracy of GNB classifier on training set:{0:.2f}"
      .format(gnb.score(X_train,y_train)))
print("Accuracy of GNB classifier on test set:{0:.2f}"
      .format(gnb.score(X_test,y_test)))
```

    Accuracy of GNB classifier on training set:0.86
    Accuracy of GNB classifier on training set:0.67
    

#### Support Vector Machine


```python
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
print("Accuracy of SVM classifier on training set:{0:.2f}"
      .format(svm.score(X_train,y_train)))
print("Accuracy of SVM classifier on test set:{0:.2f}"
      .format(svm.score(X_test,y_test)))
```

    Accuracy of SVM classifier on training set:0.61
    Accuracy of SVM classifier on test set:0.33
    

---
- 由上可看出KNN算法在这里是最好的一个模型，在训练集上的准确率百分之百，尽管我们这个数据集是非常小的。



```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
```

    [[4 0 0 0]
     [0 1 0 0]
     [0 0 8 0]
     [0 0 0 2]]
                 precision    recall  f1-score   support
    
              1       1.00      1.00      1.00         4
              2       1.00      1.00      1.00         1
              3       1.00      1.00      1.00         8
              4       1.00      1.00      1.00         2
    
    avg / total       1.00      1.00      1.00        15
    
    

#### 绘制K-NN算法的决策边界


```python
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap,BoundaryNorm
import matplotlib.patches as mpatches

X = fruits[['mass','width','height','color_score']]
y = fruits['fruit_label']
X_train,X_test,y_train, y_test = train_test_split(X,y,random_state=0)

def plot_fruit_knn(X, y, n_neighbors, weights):
    X_mat = X[['height', 'width']].as_matrix()
    y_mat = y.as_matrix()
    
    cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF','#AFAFAF'])
    cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF','#AFAFAF'])
    
    clf = KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X_mat,y_mat)
    
    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.
    
    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50
    
    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])
    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title("4-Class classification (k = %i, weights = '%s')"
           % (n_neighbors, weights))    
    plt.show()
plot_fruit_knn(X_train, y_train, 5, 'uniform')
```


![png](http://op1xwtboy.bkt.clouddn.com/output_37_0.png)



```python
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores,edgecolors=None)
plt.xticks([0,5,10,15,20])
```




    ([<matplotlib.axis.XTick at 0x21ac8ee24e0>,
      <matplotlib.axis.XTick at 0x21ac8f02f60>,
      <matplotlib.axis.XTick at 0x21ac7b565c0>,
      <matplotlib.axis.XTick at 0x21ac900a4a8>,
      <matplotlib.axis.XTick at 0x21ac900af60>],
     <a list of 5 Text xticklabel objects>)




![png](http://op1xwtboy.bkt.clouddn.com/output_38_1.png)


- 在这个特定的数据集里面，当k=5时候我们得到最高的准确率

### 总结
- 在这个文章中，我们专注于预测的准确性，我们的目的是学习一个有着好的泛化能力的模型
这样的模型使得预测的精度最大化，我们确定了最适合手头问题的机器学习算法。因此我们比较不同的算法最终选择表现最好的一个。
