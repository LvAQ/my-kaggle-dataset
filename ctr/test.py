from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss


correct = pd.read_csv('./correct_submission.csv', encoding='latin-1')
predict = pd.read_csv('./prediction_test.csv', encoding='latin-1')
correct_arr = []
predict_arr = []


for each in correct.Label:
    correct_arr.append(each)
for each in predict.Label:
    predict_arr.append([1-each, each])

loss = log_loss(correct_arr, predict_arr)