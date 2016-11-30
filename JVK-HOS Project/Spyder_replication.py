# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 22:16:19 2016

@author: James
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest
import itertools



import pandas
import numpy as np
from sklearn.grid_search import GridSearchCV
dfNotNull = pandas.read_csv('Long.csv')
#dfNotNull = dfNotNull.drop(dfNotNull.index[[218]])
dfNotNull.head()
print dfNotNull.Procedure3.value_counts()



Procedure3feat = ['Lumbar Radiofrequency', 'Cervical/Thoracic Radiofrequency', 'Facet Joint Injection', 'Cervical/Thoracic ESI'] 
Procedure3feat += ['Cervical/Thoracic Medial Branch Block', 'Spinal Cord Stimulator', 'Nerve Block', 'Lumbar Medial Branch Block']
Procedure3feat += ['Lumbar ESI', 'Joint Injection, other']
                  
Procedure4feat = ['Radiofrequency','ESI', 'Medial Branch Block', 'Injection',  'Nerve Block 2','Spinal Cord Stimulator 2']
                  
ClassFeat = ['3', 'Gender', '1r']

NonScaledContfeat  = ['Age','BMI','HealthCodesLen', 'SurgicalHistoryLen']

ScaledContfeat  = ['AgeScaled','BMIScaled','HealthCodesLenScaled', 'SurgicalHistoryLenScaled']

HealthCodes = ['M54', 'G25', 'M51', 'M50', 'M53', 'G54', 'E78', 'C177', 'D51', 'G3', 'G2', 'N28', 'I73', 'M48', 'M43', 'M41', 'M46', 'M47', 'Z79', 'F32', 'G57', 'D68', 'Q76', 'Z72', 'Y92', 'T85', 'I69', 'C182', 'M32', 'G40','F03', 'G47', 'C178', 'Y83', 'Y84', 'I50', 'C45', 'E05', 'M25', 'F12', 'F17', 'K74', 'I48', 'I42', 'C79', 'I25','E11', 'J45', 'C6', 'L40', 'E21', 'C189', 'N1', 'M96', 'M10', 'M12', 'M19', 'F41', 'Z01', 'J44', 'M79', 'J43', 'Z80','I89', 'Z85', 'Z87', 'Z86', 'Z89', 'Z88', 'C95', 'I1', 'N40', 'M84', 'M81', 'M06', 'G95', 'G90', 'R53', 'R56', 'E06', 'E07', 'K21', 'E03', 'Z96', 'Z94', 'Z95', 'Z92', 'Z90', 'Z91', 'Z98', 'E89', 'C81', 'G82', 'R26', 'Q61']

InsuranceCodes = ['BLUE CROSS','BLUE SHIELD','MEDICARE']
print dfNotNull.TotalTimeMin.mean()
sigCodesList = []
for i in HealthCodes:
    if len(dfNotNull[dfNotNull[i] == 1]) > 20 and abs(dfNotNull[dfNotNull[i] == 1].TotalTimeMin.mean() - 34.18) >=3 :
        print i + " = " + str(dfNotNull[dfNotNull[i] == 1].TotalTimeMin.mean()) + ", " + str(len(dfNotNull[dfNotNull[i] == 1]))
        sigCodesList += [i]
    
feature_cols = Procedure3feat + Procedure4feat + ClassFeat + NonScaledContfeat + sigCodesList
corrDF = dfNotNull[ NonScaledContfeat + ['TotalTimeMin']]
corrDF.corr()



print "Standar Linear Regression W/ Cross Validation - Only using Gridsearch because CV was spazzing out"

#features which gave best found score in iterative feature assesment

feature_cols =['Lumbar Radiofrequency', '3', 'Radiofrequency', 'ESI',  'Nerve Block 2','Injection','Medial Branch Block' ]#
feature_cols += ['Spinal Cord Stimulator']
tf= dfNotNull[feature_cols].drop(218)
X = tf
y = dfNotNull['TotalTimeMin'].drop(218)

print y


    
    
linreg3 = LinearRegression()

param_grid = dict()
mean2escores3 = GridSearchCV(linreg3, param_grid, cv=5, scoring='mean_squared_error').fit(X,y)
meanAbsScores3= GridSearchCV(linreg3, param_grid, cv=5, scoring='mean_absolute_error').fit(X,y)

mean2escores = mean2escores3.best_score_
print mean2escores
mean2escores = np.mean(np.sqrt(-mean2escores))
meanAbsScores= meanAbsScores3.best_score_
meanAbsScores= -meanAbsScores.mean()


print feature_cols   
print "root mean squared error"
            
print mean2escores
print "mean Abs Scores"

print meanAbsScores
print '\n'
