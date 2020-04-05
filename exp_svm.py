"""
SVM training and evaluation.
"""
#%%
import os
import numpy as np
import pandas as pd
import cv2

#%%
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data/clean/final_segmented')

#%%
train_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
train_labels = train_data.loc[:, 'lbl'].to_numpy().astype(np.int32)
train_data = train_data.iloc[:, :-1].to_numpy().astype(float).astype(np.float32)
test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
test_labels = test_data.loc[:, 'lbl'].to_numpy().astype(np.int32)
test_data = test_data.iloc[:, :-1].to_numpy().astype(np.float32)
(train_labels.shape, train_data.shape, test_data.shape, test_labels.shape)

#%%
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)

svm.trainAuto(train_data, cv2.ml.ROW_SAMPLE, train_labels)
svm.save(os.path.join(ROOT_DIR, 'models/svm1.mat'))

#%%
predictions = svm.predict(test_data)[1]
predictions = np.reshape(predictions, (len(predictions),)).astype(np.int32)
mask = np.where(predictions==test_labels, 1, 0)
correct = np.sum(mask)
accuracy = correct/len(predictions)
accuracy

# %%
import tensorflow as tf
true_tensor = tf.convert_to_tensor(test_labels, dtype=tf.int32)
pred_tensor = tf.convert_to_tensor(predictions, dtype=tf.int32)
confusion = tf.math.confusion_matrix(true_tensor, pred_tensor)
confusion

# %%
train_predictions = svm.predict(train_data)[1]
train_predictions = np.reshape(train_predictions, (len(train_predictions),)).astype(np.int32)
mask = np.where(train_predictions==train_labels, 1, 0)
correct = np.sum(mask)
train_accuracy = correct/len(train_predictions)
train_accuracy

# %%
