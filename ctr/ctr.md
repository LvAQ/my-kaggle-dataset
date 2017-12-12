# Criteo Display Advertising Challenge 

Link: [https://www.kaggle.com/c/criteo-display-ad-challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)

### 题目描述

Criteo是一家第三方广告展示公司，与世界上超过4000家电子商务公司有合作关系。Criteo共享了一周的广告展示数据，数据中提炼了13个连续特征和26个类目特征和用户是否点击了该页面的广告，希望参赛者训练出合适的模型预测用户在不同的特征下是否会点击广告。

### 先修技能

* GBDT等相关知识

### 输入格式
训练集提供了一系列的用户访问网页和点击广告的记录，l1～l13为一些计数特征，c1～c26为一些类别特征。Label表示用户是否点击广告，0为未点击，1为点击。


### 输出格式
根据测试集给出的用户访问记录，预测出用户点击某个广告的概率，输出格式如下所示：

```
Id,Predicted
60000000,0.384
63895816,0.5919
759281658,0.1934
895936184,0.9572
etc...
```

### 评价

使用Logarithmic Loss (https://www.kaggle.com/wiki/LogarithmicLoss)作为最后评判标准,公式如下：
log P(yt|yp) = yt log(yp) + (1 - yt) log(1 - yp)
可以用来表征预测值和标准值的相关性。



### 代码与数据

* train：[https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/train.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/train.csv)
* test：[https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/test.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/test.csv)
* correct_submission: [https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/correct_submission.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/correct_submission.csv)

### 完整代码

https://github.com/guestwalk/kaggle-2014-criteo


### 测评配置环境

python

```
pip install -U numpy
pip install pandas
pip install -U scikit-learn
```

测评代码

```py
from sklearn.metrics import log_loss
y_test = pd.read_csv(data_dir + "correct_submission.csv")
y_pred = pd.read_csv(data_dir + "prediction_test.csv")
accuracy = log_loss(y_test,y_pred)
```