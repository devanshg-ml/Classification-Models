import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
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



X = data.values
y = np.asarray(trainingData["label"])
clf = ExtraTreesClassifier()

results = []
for train, test in cv.split(X,y):
    y_pred = clf.fit(X[train], y[train])
    result = clf.score(X[test],y[test])
    results.append(result)

overall_perf = np.mean(results)

over_all_std = np.std(results)


final_predictions = clf.predict(X_test)

pickle.dump(clf,open("ExtraTreesModel.pkl","wb"))

### Returns mean classifier performances across all classes
print("CLASSIFIER PERFORMANCE: " + str(overall_perf))
print("CLASSIFIER STD: " + str(over_all_std))




