# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 01:08:22 2018

 
"""



#check for missing data
np.any(np.isnan(training_data), axis=0)

#use imputer to replace with median
from sklearn.preprocessing import Imputer
i=Imputer(strategy='median')
i.fit(training_data)
training_data=i.transform(training_data)
np.any(np.isnan(training_data), axis=0)
imputed_test=i.transform(test_data)


#drop all nan
mask=np.all(np.isnan(sample),axis=0)
sample[:,mask]=0

from sklearn.ensemble import RandomForestRegressor
est=RandomForestRegressor(n_estimators=10, max_features='auto',
                          max_depth=None, n_jobs=-1)
est.fit(training_data, training_results)
predicted_results=est.predict(test_data)

est.score(test_data, test_results)
est.feature_importances_

from scipy.stats import kendalltau, spearmanr
kendalltau(predicted_results, test_results)
spearmanr(predicted_results, test_results)

# optimization
from itertools import product
params = [(100,200), (0.3,0.5)]
best_score=float("-inf")
for n,f in product(*params):
    est = RandomForestRegressor(oob_score=True, n_estimators=n, max_features=f)
    est.fit(training_data, training_results)
    if est.oob_score_ > best_score:
        best_score, best_est=est.oob_score_, est

export_graphviz