#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 01:12:13 2019

@author: anibaljt
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import pickle

## 5 Fold cross validation
cv = KFold(n_splits=5)

### Reads in training data from csv
trainingData =pd.read_csv(os.path.expanduser('~/Downloads/training.csv'))
X_test = pd.read_csv(os.path.expanduser('~/Downloads/testing.csv'))
                                        
### Excludes incidces and labels
data = trainingData.T.iloc[1:len(trainingData)]
data = data.T
test_data = X_test.T.iloc[1:len(X_test)].values
test_data = test_data.T

### OneVsRest classifier - one classifier per class,
### fits one class against all other classes
clf = RandomForestClassifier(n_estimators = 200)

X = data.values
y = np.asarray(trainingData["label"])

results = []
for train, test in cv.split(X,y):
    y_pred = clf.fit(X[train], y[train])
    result = clf.score(X[test],y[test])
    print(result)
    results.append(result)

over_all_std = np.std(results)
overall_perf = np.mean(results)

final_predictions = clf.predict(X_test)

pickle.dump(clf,open("RFCModel.pkl","wb"))

### Returns mean classifier performances across all classes
print("CLASSIFIER PERFORMANCE: " + str(overall_perf))
print("CLASSIFIER STD: " + str(over_all_std))



