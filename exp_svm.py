"""
SVM training and evaluation.
"""
# %%
import pickle
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import recall_score as rcl
from sklearn.metrics import precision_score as psn
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# %%
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data/clean/final_segmented')

# %%
train_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
train_labels = train_data.loc[:, 'lbl'].to_numpy().astype(np.int32)
train_labels = MultiLabelBinarizer().fit_transform(train_labels.reshape(-1, 1))
train_labels = np.delete(train_labels, 1, 1)
train_data = train_data.iloc[:, :-
                             1].to_numpy().astype(float).astype(np.float32)
test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
test_labels = test_data.loc[:, 'lbl'].to_numpy().astype(np.int32)
test_labels = MultiLabelBinarizer().fit_transform(test_labels.reshape(-1, 1))
test_labels = np.delete(test_labels, 1, 1)
test_data = test_data.iloc[:, :-1].to_numpy().astype(np.float32)
(train_labels.shape, train_data.shape, test_data.shape, test_labels.shape)

# %%
svm = make_pipeline(StandardScaler(), OneVsRestClassifier(LinearSVC()))
svm.fit(train_data, train_labels)

# %%
test_pred = svm.predict(test_data)
train_pred = svm.predict(train_data)

# %%

# %%
faw_acc_test = acc(test_labels[:, 0], test_pred[:, 0])
faw_acc_train = acc(train_labels[:, 0], train_pred[:, 0])
zinc_acc_test = acc(test_labels[:, 1], test_pred[:, 1])
zinc_acc_train = acc(train_labels[:, 1], train_pred[:, 1])

faw_psn_test = psn(test_labels[:, 0], test_pred[:, 0])
zinc_psn_test = psn(test_labels[:, 1], test_pred[:, 1])

faw_rcl_test = rcl(test_labels[:, 0], test_pred[:, 0])
zinc_rcl_test = rcl(test_labels[:, 1], test_pred[:, 1])

faw_f1_test = f1(test_labels[:, 0], test_pred[:, 0])
zinc_f1_test = f1(test_labels[:, 1], test_pred[:, 1])

faw_auc_test = auc(test_labels[:, 0], test_pred[:, 0])
zinc_auc_test = auc(test_labels[:, 1], test_pred[:, 1])

# %%
history = {
    "faw_acc": faw_acc_test,
    "zinc_acc": zinc_acc_test,
    "faw_psn": faw_psn_test,
    "zinc_psn": zinc_psn_test,
    "faw_rcl": faw_psn_test,
    "zinc_rcl": zinc_rcl_test,
    "faw_f1": faw_f1_test,
    "zinc_f1": zinc_f1_test,
    "faw_auc": faw_auc_test,
    "zinc_auc": zinc_auc_test
}

# %%

with open("/home/ngari/Dev/projectzeamays/models/svm.pkl", 'wb') as file:
    pickle.dump(svm, file)

# %%
with open("/home/ngari/Dev/projectzeamays/models/svm.pkl", 'rb') as file:
    pickle_model = pickle.load(file)

test_pred = pickle_model.predict(test_data)
train_pred = pickle_model.predict(train_data)

# %%
acc(test_labels, test_pred)

# %%
