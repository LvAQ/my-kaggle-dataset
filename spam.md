# SMS Spam Collection Dataset描述

## 是垃圾短信，不是垃圾短信，这是一个问题。

Link: [https://www.kaggle.com/uciml/sms-spam-collection-dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

### 题目描述
下面有一个数据集, 它包括了5574条英文短信，短信内容由长短不一的几句话组成。每条短信都标注好了是否为垃圾短信，通过该训练集训练处一个分类器，预测短信内容是否为垃圾短信。

### 先修技能

* 会使用svm或者bayes等分类器
* 原始数据集可参见链接：http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/


### 输入格式
数据文件train.csv包含5574英文短信。
每条短信占一行，每行由两列构成。第一列是短信相应的标签`label`，如果是`ham`表示为非垃圾短信，如果是`spam`表示为垃圾短信。第二列是短信的具体内容，由长短不一的几句英文组成。如下所示：

```
Label,Text
ham,Aight should I just plan to come up later tonight?
ham,Was the farm open?
ham,"I sent my scores to sophas and i had to do secondary application for a few schools. I think if you are thinking of applying, do a research on cost also. Contact joke ogunrinde, her school is one me the less expensive ones"
etc...
```

数据文件test.csv每行由两列构成。第一列是短信相应的Id`SmsId`,第二列是短信的具体内容，由长短不一的几句英文组成。如下所示：

```
SmsId,Text
4456,Aight should I just plan to come up later tonight?
690,Was the farm open?
944,"I sent my scores to sophas and i had to do secondary application for a few schools. I think if you are thinking of applying, do a research on cost also. Contact joke ogunrinde, her school is one me the less expensive ones"
etc...
```


### 输出格式
您的提交文件应采用以下格式：对于测试集中的每条短信，输出一行，其中包含SmsId和您预测其是否为垃圾短信的结果，不是为ham、是为spam。 例如，如果您预测第一条短信是垃圾短信，第二条不是，那么您的提交文件将如下所示：

```
SmsId,Label
1,spam
2,ham
etc...
```

### 评价

使用[准确率(accuracy)](https://www.zhihu.com/question/19645541)作为最后评判标准。

```
TP，True Positive，将正类预测为正类的数目

FP，False Positive，将负类预测为正类数

TN，True Negative，将负类预测为负类数

FN，False Negative，将正类预测为负类数
```
$$ Accuracy=\frac{TP+TN}{TP+FN+FP+TN} $$


### 代码与数据

* train：[https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/train.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/train.csv)
* test：[https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/test.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/test.csv)
* correct_submission: [https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/correct_submission.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/correct_submission.csv)

### 完整代码

https://www.kaggle.com/jiuzhang/ninechapter-spam-ham-self/notebook


### 测评配置环境

python

```
pip install -U numpy
pip install pandas
pip install -U scikit-learn
```

测评代码

```py
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

y_test = pd.read_csv("./correct_submission.csv").sort_values('SmsId')
y_pred = pd.read_csv("./correct_submission.csv").sort_values('SmsId')

accuracy = accuracy_score(y_test.Label,y_pred.Label)
```