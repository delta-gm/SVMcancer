# sklearn isolation forest with credit card 

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.ensemble import IsolationForest 

creditcard = pd.read_csv('creditcard.csv')
print(creditcard[:3])
# creditcard['scores'] = pd.Series(np.ones)
# creditcard['anomaly'] = pd.Series(np.ones)
X = creditcard.sample(frac=0.1,random_state=1).reset_index(drop=True) # df[~((df['Rating']>8) | (df['Metascore']>80))]
print(X[:3])

# Checking to make sure outlier fraction is same as original: should be .001723
fraudulent_transactions = X[X['Class']==1]
print('Fraudulent Transactions:', len(fraudulent_transactions))
legit_transactions = X[X['Class']==0]
outlier_fraction = len(fraudulent_transactions)/float(len(legit_transactions))
print(outlier_fraction)

# define test train split:
X_train = X.sample(frac=0.80)
# print(X_train[:3])
X_test = X.sample(frac=0.20)
model = IsolationForest(n_estimators = 50, max_samples=100, contamination = (100*outlier_fraction), max_features = 1)   #random_state=1
model.fit(X_train)
yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)
yhat_outliers = model.predict(fraudulent_transactions)
# anomalies = X_test.loc[X_test['anomaly']==-1]
# anomaly_index = list(anomalies.index)
# print('\n', anomalies, '\n')
X['scores'] = model.decision_function(X)
model.fit(X)
X['anomaly'] = model.predict(X)
print('training yhat: \n', yhat_train[:40], '\n test yhat:', yhat_test[:40], '\n fraudulent row prediction rate: \n', yhat_outliers, 'number of fraudulent transactions identified: \n', len(yhat_outliers))
print(X.shape)

correct_detections = X[(X['anomaly']==-1) & (X['Class']==1)]
print(correct_detections[:10])
print(len(correct_detections))

# creditcard['scores'] = model.decision_function(creditcard)
# model.fit(creditcard)
# creditcard['anomaly'] = model.predict(creditcard)

# results = creditcard[(creditcard['anomaly']==-1) & (creditcard['Class']==1)]
# print(results)











# scratch shit
# sklearn_score_anomalies = abs(model.score_samples(X))
# print(sklearn_score_anomalies)


# ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']
# ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28']



# Define and Fit Model

# We'll create a model variable and instantiate the IsolationForest class. We are passing the values of four parameters to the Isolation Forest method, listed below.

# Number of estimators: n_estimators refers to the number of base estimators or trees in the ensemble, i.e. the number of trees that will get built in the forest. 
# This is an integer parameter and is optional. The default value is 100.

# Max samples: max_samples is the number of samples to be drawn to train each base estimator. If max_samples is more than the number of samples provided, all samples 
# will be used for all trees. The default value of max_samples is 'auto'. If 'auto', then max_samples=min(256, n_samples)

# Contamination: This is a parameter that the algorithm is quite sensitive to; it refers to the expected proportion of outliers in the X set. This is used when 
# fitting to define the threshold on the scores of the samples. The default value is 'auto'. If ‘auto’, the threshold value will be determined as in the original paper of 
# Isolation Forest.

# Max features: All the base estimators are not trained with all the features available in the Xset. It is the number of features to draw from the total features 
# to train each base estimator or tree.The default value of max features is one.

# model = IsolationForest(n_estimators = 50, max_samples = 28, contamination = float(0.1), max_features=1.0)
# model.fit(X[['Class']])