# SMS Spam Collection Dataset描述

## 是垃圾邮件，不是垃圾邮件，这是一个问题。

Link: [https://www.kaggle.com/uciml/sms-spam-collection-dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

### 题目描述

给一条英文短信，短信内容由长短不一的几句话组成，试判断该短信是否为垃圾短信。

### 先修技能

* 会svm或者bayes等分类器的使用
* 原始数据集可参见链接：http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/


### 输入格式
数据文件spam.csv包含5574英文短信。
每条短信占一行，每行由两列构成。第一列为label，内容为ham表示非垃圾短信，内容为spam表示为垃圾短信。第二列为短信内容，由长短不一的几句话组成。


### 输出格式
您的提交文件应采用以下格式：对于测试集中的每条短信，输出一行，其中包含SmsId和您预测其是否为垃圾短信的结果，不是为ham、是为spam。 例如，如果您预测第一条短信是垃圾短信，第二条不是，那么您的提交文件将如下所示：

```
SmsId,Label
1,spam
2,ham
etc...
```

### 评价

使用 accuracy 最后评判标准


### 代码与数据

* train：[https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/train.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/train.csv)
* test：[https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/test.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/spam/test.csv
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
from sklearn.metrics import accuracy_score
y_test = pd.read_csv(data_dir + "correct_submission.csv")
y_pred = pd.read_csv(data_dir + "prediction_test.csv")
accuracy = accuracy_score(y_test,y_pred)
```