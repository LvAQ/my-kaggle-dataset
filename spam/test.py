import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

y_test = pd.read_csv("./correct_submission.csv").sort_values('SmsId')
y_pred = pd.read_csv("./correct_submission.csv").sort_values('SmsId')

accuracy = accuracy_score(y_test.Label,y_pred.Label)
