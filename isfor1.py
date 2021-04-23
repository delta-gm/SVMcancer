# full working with multiple pcas


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pprint import pprint
import seaborn as sns
import pdb
sns.set_style(style="whitegrid")
from matplotlib import rcParams
# plt.style.use('fivethirtyeight')
# rcParams['axes.labelsize'] = 14
# rcParams['xtick.labelsize'] = 12
# rcParams['ytick.labelsize'] = 12
# rcParams['text.color'] = 'k'
# rcParams['figure.figsize'] = 16,8

import time
startTime = time.time()

# read in 1% of cc csv as dataframe, drop index and a shitload of columns.
sampleSize = 0.5
df = pd.read_csv('creditcard.csv')
X = df.sample(frac = sampleSize, random_state=1).reset_index(drop = True)
# 'Time','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class'
toDrop = ['Time','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Class']
X.drop(toDrop, inplace = True, axis = 1)


# fraudClass is a df with original index and class as columns.
fraudClass = df.loc[:,'Class'].sample(frac=sampleSize, random_state=1).reset_index(drop = False)  
fraudClass.rename(columns = {'index':'rawIndex'}, inplace = True)
fraudClass.rename(columns = {'Class':'trueClass'}, inplace = True)
print(fraudClass[:5])
positiveCases = fraudClass[fraudClass['trueClass'] == 1].index                              
print('anomaly locations in X : \n', positiveCases)
outlierFraction = len(positiveCases)/fraudClass.shape[0]                                    
# print(' outlier fraction : \n', outlierFraction)


def select_feature(data): 
    return random.choice(data.columns)

def select_value(data,feat):
    mini = data[feat].min()
    maxi = data[feat].max()
    return (maxi-mini)*np.random.random()+mini

def split_data(data, split_column, split_value):                    
    data_below = data[data[split_column] <= split_value]
    data_above = data[data[split_column] >  split_value]
    
    return data_below, data_above

def terminalNode(data, counter):
    label_column = data.values[:,-1]
    label_column = np.append(label_column, np.array([0]), axis = 0)
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    counts = np.append(counts_unique_classes, np.array([0]), axis = 0)
    index = counts.argmax()
    if index == 0:
        ext_node = label_column[index]
        return ext_node
    else:
        ext_node = unique_classes[index]
        return ext_node
    

def isolationTree(data, counter = 0, max_depth = 50):
    if (counter == max_depth) or data.shape[0]<=1:
        ext_node = terminalNode(data, counter)
        return ext_node
    
    else:
        counter +=1                                                             # Counter
        split_column = select_feature(data)                                     # select feature
        split_value = select_value(data,split_column)                           # select value
        data_below, data_above = split_data(data,split_column,split_value)      # split data
        question = "{} <= {}".format(split_column, split_value)                 # instantiate sub-tree     
        sub_tree = {question: []}
        below_answer = isolationTree(data_below, counter, max_depth=max_depth)        # Recursive part
        above_answer = isolationTree(data_above, counter, max_depth=max_depth)
        
        if below_answer == above_answer:
            sub_tree = below_answer
        else:
            sub_tree[question].append(below_answer)
            sub_tree[question].append(above_answer)
        
        return sub_tree


def isolationForest(df,n_trees=5, max_depth=50, subspace=1024):
    forest = []
    for i in range(n_trees):
        # Sample the subspace
        if subspace<=1:
            df = df.sample(frac=subspace)
        else:
            df = df.sample(subspace)
        
        # Fit tree
        tree = isolationTree(df,max_depth=max_depth)
        
        # Save tree to forest
        forest.append(tree)
    
    return forest



def pathLength(example,iTree,path=0,trace=False):
    path=path+1
    question = list(iTree.keys())[0]
    
    feature_name, comparison_operator, value = question.split()
    # ask question
    if example[feature_name].values <= float(value):
        answer = iTree[question][0]
    else:
        answer = iTree[question][1]

    # base case
    if not isinstance(answer, dict):
        return path
    
    # recursive part
    else:
        residual_tree = answer
        return pathLength(example, residual_tree,path=path)

    return path


# tree = isolationTree(X.head(50),max_depth=3)

# ins = X.iloc[[82]]
# path1 = pathLength(ins,tree)
# print('Path: ', path1)

# ins2 = X.iloc[[1]]
# path2 = pathLength(ins2,tree)
# print('Path: ', path2)


# def makeline(data,example,iTree,path=0,line_width=1):
#     #line_width = line_width +2
#     path=path+1
#     question = list(iTree.keys())[0]
#     feature_name, comparison_operator, value = question.split()
#     print(question)
    
#     # ask question
#     if example[feature_name].values <= float(value):
#         answer = iTree[question][0]
#         data = data[data[feature_name] <= float(value)]
#     else:
#         answer = iTree[question][1]
#         data = data[data[feature_name] > float(value)]
        
#     if feature_name == 'V1':
#         plt.hlines(float(value),xmin=data.V1.min(),xmax=data.V1.max(),linewidths=line_width)
#     else:
#         plt.vlines(float(value),ymin=data.V2.min(),ymax=data.V2.max(),linewidths=line_width)
             
#     # base case
#     if not isinstance(answer, dict):
#         return path
    
#     # recursive part
#     else:
#         if feature_name == 'V1':
#             plt.hlines(float(value),xmin=data.V1.min(),xmax=data.V1.max(),linewidths=line_width)
#         else:
#             plt.vlines(float(value),ymin=data.V2.min(),ymax=data.V2.max(),linewidths=line_width)
#         residual_tree = answer
#         return makeline(data,example, residual_tree,path=path,line_width=line_width)
    
#     return path


def evaluate_instance(instance,forest):
    paths = []
    for tree in forest:
        paths.append(pathLength(instance,tree))
    return paths

def c_factor(n) :
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

def anomaly_score(data_point,forest,n):
    E = np.mean(evaluate_instance(data_point,forest))
    c = c_factor(n)
    return 2**-(E/c)


# ntrees = 20, maxdepth = 100, subspace = 256
iForest = isolationForest(X, n_trees=60, max_depth=20, subspace=300)
# success 1: 20 20 100, 1 correct detection
# success 2: 20 20 300, 2 correct detections

an = []
for i in range(X.shape[0]):
    an.append(anomaly_score(X.iloc[[i]],iForest,1024))

X.loc[:,'anomalyScore'] = pd.Series(an, index = X.index)


# instance_depth_plot(X.sample(1),X.head(1),iForest)
outlier1 = evaluate_instance(X.iloc[[82]],iForest)
outlier2 = evaluate_instance(X.iloc[[345]],iForest)
outlier3 = evaluate_instance(X.iloc[[489]],iForest)

normal1 = evaluate_instance(X.iloc[[60]],iForest)
normal2 = evaluate_instance(X.iloc[[164]],iForest)
normal3 = evaluate_instance(X.iloc[[2045]],iForest)

print(outlier1)
print(outlier2)
print(outlier3)
print(normal1)
print(normal2)
print(normal3)

print('outlier average1: \n', np.mean(outlier1))
print('outlier average2: \n', np.mean(outlier2))
print('outlier average3: \n', np.mean(outlier3))

print('normal1 average: \n', np.mean(normal1))
print('normal2 average: \n', np.mean(normal2))
print('normal3 average: \n', np.mean(normal3))


print('Anomaly score for outlier1:',anomaly_score(X.iloc[[82]],iForest,1024))
print('Anomaly score for outlier2:',anomaly_score(X.iloc[[345]],iForest,1024))
print('Anomaly score for outlier3:',anomaly_score(X.iloc[[489]],iForest,1024))

print('Anomaly score for normal:',anomaly_score(X.iloc[[60]],iForest,1024))
print('Anomaly score for normal:',anomaly_score(X.iloc[[164]],iForest,1024))
print('Anomaly score for normal:',anomaly_score(X.iloc[[2045]],iForest,1024))



def classifier(anomalyScore):                   
    num_anom = len(positiveCases) + 3              
    an.sort()                         
    topScores = an[-(num_anom):]
    thresholdValue = min(topScores)
    if anomalyScore >= thresholdValue:
        return 1
    else:
        return 0

X['detectionClass'] = X.apply(lambda x: classifier(x['anomalyScore']), axis = 1)
predictedAnomalies = X[X['detectionClass']==1]
detection = predictedAnomalies.drop(['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14'], axis = 1, inplace = False)
print('\n Detected Anomalies: \n', detection)

dfPosIdx = []
for i in positiveCases:
    # print(fraudClass.iloc[i])
    finalidx = int(fraudClass.iloc[i]['rawIndex'])
    dfPosIdx.append(df.iloc[finalidx])

dfTrue = pd.DataFrame(dfPosIdx, columns=['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class'])
trueDrop = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14']
dfTrue.drop(trueDrop, inplace = True, axis = 1)
dfTrue['0.01 index'] = pd.Series(positiveCases, index = dfTrue.index)
print('\n True Anomalies: \n', dfTrue)

finalResults = pd.merge(X, fraudClass, on = [X.index])
finalDrop = ['key_0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14']
finalResults.drop(finalDrop, inplace = True, axis = 1)
results = finalResults[finalResults['detectionClass']==1]
results = results[['rawIndex', 'anomalyScore', 'detectionClass', 'trueClass', 'Amount']]
# print('\n Final Results: \n', results)

def judge(detectionClass, trueClass):
    if detectionClass == trueClass:
        return 'correct'
    else:
        return 'falsePositive'

results['Result'] = results.apply(lambda x: judge(x['detectionClass'],x['trueClass']), axis = 1)
print('\n Final Results: \n', results)

correctAnswers = results[results['Result'] == 'correct'].index
print('Correct detections : \n', correctAnswers)
correctFraction = len(correctAnswers)/results.shape[0] 
print(correctFraction)

executionTime = (time.time() - startTime)
print('\n Execution time in seconds: \n' + str(executionTime) + '\n')



