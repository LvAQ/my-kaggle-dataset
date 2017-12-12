\# Facial Keypoints Detection

  


\#\# 你能教会计算机识别五大洲人种的眼耳鼻喉吗？

  


Link:

\[

https://www.kaggle.com/c/facial-keypoints-detection

\]

\(

https://www.kaggle.com/c/facial-keypoints-detection

\)

  


\#\#\# 题目描述

  


从人物头像的96x96像素的灰度照片中找出代表面部器官位置的关键点（keypoints）的坐标，关键点包括左右眼中心等，共15个。

  


\#\#\# 先修技能

  


\*

CNN的相关知识。

  


\#\#\# 输入格式

训练集给了大约5000个人物头像的灰度图片，像素96x96，灰度0-255，图片数据的矩阵被整理成一维向量，并附有每个头像15个关键点（keypoints）的位置坐标（x轴y轴）。

  


  


\*

\`labeledTrainData.csv\`

是有Label的训练集，包含20000行评论数据，包括评论id、评论文本、情感（0代表消极，1代表积极）。

\*

\`testData.csv\`

是一个无Label的测试集，包含5000行评论数据，包括评论id、评论文本，没有对应情感。

\*

\`unlabeledTrainData\`

是额外的50000行无Label数据，包括评论id和评论文本。

\*\*可用来进行文本特征提取或半监督学习。\*\*

\*

\`sampleSubmission\`

是一个提交格式的样例。

  


\#\#\# 输出格式

  


根据测试集给出的头像图片数据，预测出每个人物头像的关键点的位置坐标，输出格式如下所示：

  


如下所示：

  


\`\`\`

RowId,ImageId,FeatureName,Location

1,1,left\_eye\_center\_x,?

2,1,left\_eye\_center\_y,?

3,1,right\_eye\_center\_x,?

4,1,right\_eye\_center\_y,?

etc...

\`\`\`

  


\#\#\# 评价

  


使用RMSE作为最后评判标准

RMSE，方均根偏移\(

\*\*root-mean-square deviation\*\*

\)或方均根差\(

\*\*root-mean-square error\*\*

\)是一种常用的测量数值之间差异的量度。具体公式如下：

  


  


!\[\]

\(

http://www.statisticshowto.com/wp-content/uploads/2016/10/root-mean-square-error.png

\)

  


  


\#\#\# 代码与数据

  


\*

train：

\[

https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip

\]

\(

https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip

\)

\*

test：

\[

https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip

\]

\(

https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip

\)

\*

samplesubmission:

\[

https://www.kaggle.com/c/facial-keypoints-detection/download/SampleSubmission.csv

\]

\(

https://www.kaggle.com/c/facial-keypoints-detection/download/SampleSubmission.csv

\)

  


\#\#\# 完整代码

  


\*

误差 2.13 ：

\[

http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

\]

\(

http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

\)

  


  


  


\#\#\# 测评配置环境

  


python

  


\`\`\`

pip install -U numpy

pip install pandas

pip install -U scikit-learn

\`\`\`

  


测评代码

  


\`\`\`py

from

sklearn.metrics

import

mean\_squared\_error

y\_test = pd.read\_csv\(data\_dir +

"correct\_submission.csv"

\)

y\_pred = pd.read\_csv\(data\_dir +

"prediction\_test.csv"

\)

accuracy = mean\_squared\_error\(y\_test,y\_pred\)

\`\`\`

