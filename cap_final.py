# capstone evaluation final task
'''
permalink:
https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/bd3a10ca-4d42-428b-86b2-21a8f2565737/view?access_token=0ffa89b4f51d5900b7cc5a96bbf480dfb657ec16afbdd2933b2e3d07614047cf
'''

from sklearn.tree import DecisionTreeClassifier
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

test_df = pd.read_csv('loan_test.csv')
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1],inplace=True)
test_feature = test_df[['Principal','terms','age','Gender','weekend']]
test_feature = pd.concat([test_feature,pd.get_dummies(test_df['education'])], axis=1)
test_feature.drop(['Master or Above'], axis = 1,inplace=True)
test_feature.head()
X = test_feature
y = test_df['loan_status'].values
X = preprocessing.StandardScaler().fit(X).transform(X)

loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = 20)
loanTree.fit(X,y)
dt_yhat = loanTree.predict(X)
print("DecisionTree's Accuracy: %5.2f" % metrics.accuracy_score(y, dt_yhat))
print("DT Jaccard Score: %5.2f" % jaccard_score(y, dt_yhat))
print("DT F1 Score: %5.2f" % f1_score(y, dt_yhat, average='weighted'))

from sklearn.neighbors import KNeighborsClassifier
k = 6
neigh = KNeighborsClassifier(n_neighbors = k).fit(X,y)
k_yhat = neigh.predict(X)
neigh
k_yhat[0:5]
print("KNN Jaccard: %5.2f" % jaccard_score(y, k_yhat))
print("KNN F1: %5.2f" % f1_score(y, k_yhat, average='weighted'))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X,y)
LR
yhat = LR.predict(X)
yhat_prob = LR.predict_proba(X)
yhat
yhat_prob
print("LR Jaccard: %5.2f" % jaccard_score(y, yhat))
print("LR F1: %5.2f" % f1_score(y, yhat, average='weighted'))
print("LR Log %5.2fLoss: " % log_loss(y, yhat_prob))

from sklearn import svm
clf = svm.SVC(kernel='rbf', gamma =0.1)
clf.fit(X, y) 
svm_yhat = clf.predict(X)

print("SVM Jaccard: %5.2f" % jaccard_score(y, svm_yhat))
print("SVM F1: %5.2f" % f1_score(y, svm_yhat, average='weighted'))


'''
| Algorithm          | Jaccard | F1-score | LogLoss |
|--------------------|---------|----------|---------|
| KNN                | .907    | .900     | NA      |
| Decision Tree      | .963    | .962     | NA      |
| SVM                | .870    | .854     | NA      |
| LogisticRegression | .777    | .782     | 7.67    |
'''