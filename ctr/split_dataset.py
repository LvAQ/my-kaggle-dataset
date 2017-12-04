from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


df = pd.read_csv('./ctr.csv', encoding='latin-1')

# print (df)

data_train, data_test = train_test_split(
    df,
    test_size=0.2, 
    random_state=0)

data_train.to_csv('train.csv', index=False)

labels_test = data_test.loc[:,['Id', 'Label']]
data_test = data_test.loc[:, ['Id', 'I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26']]

labels_test.to_csv('correct_submission.csv', index=False)
data_test.to_csv('test.csv', index=False)
