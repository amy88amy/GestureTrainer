import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

root_dir = os.getcwd() + "/Dataset_HAR"
df_g=pd.DataFrame()
df_a=pd.DataFrame()
df=pd.DataFrame()
label = pd.DataFrame(columns=['label'])
for subdir, dirs, files in os.walk(root_dir):
  for file in files:
    path = os.path.join(subdir, file)
    l = path.split('/')
    if(l[-1]=="gyroscope.csv"):
      df_csv = pd.read_csv(path)
      df_csv=df_csv.drop('time',1)
      df_csv.columns=['gx','gy','gz']
      del_len=int(len(df_csv)*0.1)
      df_csv = df_csv.iloc[int(del_len/2):-int(del_len/2)]
      df_g= pd.concat([df_g,df_csv])
 
      
    if(l[-1]=="accelerometer.csv"):
      df_csv = pd.read_csv(path)
      df_csv=df_csv.drop('time',1)
      df_csv.columns=['ax','ay','az']
      del_len=int(len(df_csv)*0.1)
      df_csv = df_csv.iloc[int(del_len/2):-int(del_len/2)]
      df_a= pd.concat([df_a,df_csv])
      for k in range(0, len(df_csv)):
        label.loc[len(label.index), 'label'] = l[-3]
df_g=df_g.reset_index(drop=True)
df_a=df_a.reset_index(drop=True)
df=pd.concat([df_a,df_g],axis=1)

df['label']=label

window_size = 30
stride = 15
shape = df.shape
unique_labels = df.label.unique()

column_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label']
final_column_names = ['max', 'may', 'maz', 'mgx', 'mgy', 'mgz', 'sax', 'say', 'saz', 'sgx', 'sgy', 'sgz', 'label']

data = pd.DataFrame(columns=final_column_names)

for activity in unique_labels:
  curr_df = df[df.label == activity]
  in_size = curr_df.shape[0]
  out_size = int(math.floor((in_size - window_size)/stride) + 1)

  for i in range(out_size):
    new_mean_row = curr_df.iloc[i*stride : (i*stride) + window_size].mean()
    new_sd_row = curr_df.iloc[i*stride : (i*stride) + window_size].std()
    new_row = new_mean_row.append(new_sd_row)
    final_row = new_row.append(pd.Series([activity]), ignore_index=True)
    final_row.index = final_column_names
    data = data.append(final_row, ignore_index=True)

data = data.sample(frac=1)
labels = data.pop('label')
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=12, verbose=2, n_iter_no_change=50)
history = clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

precision = precision_score(y_test, predictions, labels=unique_labels, average=None)
print("Precision: " , (precision * 100.0))

recall = recall_score(y_test, predictions, labels=unique_labels, average=None)
print("Recall: ", (recall * 100.0))

filename = 'HARClassifier.sav'
pickle.dump(clf, open(filename, 'wb'))
