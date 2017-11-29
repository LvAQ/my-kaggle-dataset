from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


df = pd.read_csv('./spam.csv', encoding='latin-1')

data_train, data_test, labels_train, labels_test = train_test_split(
    df.v2,
    df.v1, 
    test_size=0.2, 
    random_state=0)

data_train.to_csv('train.csv')
data_test.to_csv('test.csv')
labels_test.to_csv('correct_submission.csv')