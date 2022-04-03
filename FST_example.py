# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:00:24 2022

@author: jandr
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder

from fuzzy_random_survival_forest import RandomSurvivalForest
from fuzzy_random_survival_forest import concordance_index 


X, y = load_gbsg2()

grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

X_no_grade = X.drop("tgrade", axis=1)
Xt = OneHotEncoder().fit_transform(X_no_grade)
Xt = np.column_stack((Xt.values, grade_num))


feature_names = X_no_grade.columns.tolist() + ["tgrade"]

scaler=MinMaxScaler()
Xt = scaler.fit_transform(Xt)
Xt=np.round(Xt, 4)
random_state = 0
deaths=3
samples=20
arboles=1
dicoto=[2,3,7]

X_trains, X_tests, y_trains, y_tests = train_test_split(
    Xt, y, test_size=0.25, random_state=random_state)


y_train=pd.DataFrame(y_trains)
y_train=y_train[['time','cens']]

y_test=pd.DataFrame(y_tests)
y_test=y_test[['time','cens']]


X_test=pd.DataFrame(X_tests)
X_train=pd.DataFrame(X_trains)

y_train['time']=y_train.time.astype('int')

########################################################################################
# FST 
########################################################################################

rsf = RandomSurvivalForest(n_estimators=arboles, random_state=random_state,dicoto, unique_deaths=deaths, min_leaf=samples, n_jobs=1)
rsf.fit(X_train,y_train)
y_pred=rsf.predict(X_test)
c_val_fuzzy = concordance_index(y_test.time, y_pred, y_test.cens)

##################################################################################
# C-Index RST (crisp algorithm)
##################################################################################

from random_survival_forest import RandomSurvivalForest as J_RSF

rsforest = J_RSF(n_estimators=arboles,unique_deaths=deaths, min_leaf=samples,random_state=random_state)

ytrainj=y_train[['cens','time']]
rsforest.fit(X_train,ytrainj)     
y_pred = rsforest.predict(X_test)
c_val_crisp = concordance_index(y_test.time, y_pred, y_test.cens)


print('Results','FST_cindex = ',c_val_fuzzy,'RST_cindex = ',c_val_crisp)
#Results FST_cindex =  0.6120418259500787 RST_cindex =  0.5501461659545761