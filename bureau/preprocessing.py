# -*- coding: utf-8 -*-
"""
Created on Tue May 12 07:33:03 2020

@author: Иван
"""

import pandas as pd
import numpy as np
# грузим данные
train = pd.read_csv('application_train.csv',  delimiter = ',')
#Выделяем целевую величину
target = train[['TARGET','SK_ID_CURR']]
train.drop(['TARGET'], axis = 1, inplace = True)
test = pd.read_csv('application_test.csv', delimiter = ',')
bureau =  pd.read_csv('bureau.csv', delimiter = ',')
# добавим целевую величину в данные о кредитной истории
df_all = pd.merge_ordered(bureau, target, on='SK_ID_CURR', how='left') 
df_all.dropna(subset = ['TARGET'], inplace = True)
#Удаляем данные по валюте
df_all.drop(['CREDIT_CURRENCY', 'SK_ID_BUREAU', 'AMT_ANNUITY'],axis=1, inplace=True)
#Заменяем данные на числовые
df_all.CREDIT_ACTIVE.replace({'Active':0,'Closed':1, 'Sold': 1, 'Bad debt' :-1}, inplace = True)
df_all.astype({'CREDIT_ACTIVE':'float64'})
#Бинарное кодирование типа кредита 
df_all = pd.get_dummies(df_all, columns = ['CREDIT_TYPE'])
#Усредняем  фичи которые усредняются 
df_all_mean = df_all.groupby(['SK_ID_CURR']).mean()
#добавляем фичу число закрытых кредитов также штрафуя за просроченные кредиты 
df = df_all.groupby(['SK_ID_CURR']).sum()
df_all_mean['COUNT_CLOSED_CREDIT'] = df_all.groupby(['SK_ID_CURR']).sum()['CREDIT_ACTIVE']
df_all_mean['COUNT_LATE_CREDIT'] = df_all.groupby(['SK_ID_CURR']).count()['CREDIT_ACTIVE']
#Работаем с миссингами

for i in range(0, df_all_mean.shape[1]):
    
    df_all_mean[df_all_mean.columns[i]].fillna(df_all_mean[df_all_mean.columns[i]].mean(), inplace=True  )
# 
target = df_all_mean['TARGET']
df_all_mean.drop(['TARGET'], axis=1 , inplace =True)
# обучаем модель предсказывать хорошая/плохая кредитная история основываясь на том 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
forest = RandomForestClassifier(n_estimators = 1000, max_depth = 7, max_features = 0.5,class_weight= 'balanced')
# score = cross_val_score(forest, df_all_mean, target,cv=2, scoring = 'recall')
forest.fit(df_all_mean, target)
# формируем данные для  предикта
bureau.drop(['CREDIT_CURRENCY', 'SK_ID_BUREAU', 'AMT_ANNUITY'],axis=1, inplace=True)
bureau.CREDIT_ACTIVE.replace({'Active':0,'Closed':1, 'Sold': 1, 'Bad debt' :-1}, inplace = True)
bureau.astype({'CREDIT_ACTIVE':'float64'})
bureau = pd.get_dummies(bureau, columns = ['CREDIT_TYPE'])
bureau_mean = bureau.groupby(['SK_ID_CURR']).mean()
bureau_mean['COUNT_CLOSED_CREDIT'] = bureau.groupby(['SK_ID_CURR']).sum()['CREDIT_ACTIVE']
bureau_mean['COUNT_LATE_CREDIT'] = bureau.groupby(['SK_ID_CURR']).count()['CREDIT_ACTIVE']
#Работаем с миссингами
for i in range(0, bureau_mean.shape[1]):
    bureau_mean[bureau_mean.columns[i]].fillna(bureau_mean[bureau_mean.columns[i]].mean(), inplace=True  )

# Число активных кредитов
#act =  bureau[bureau['CREDIT_ACTIVE']==0].groupby(['SK_ID_CURR']).count()['CREDIT_ACTIVE']

# Число  плохих кредитов
#bad =  bureau[bureau['CREDIT_ACTIVE']==-1].groupby(['SK_ID_CURR']).count()['CREDIT_ACTIVE']
#bad[bad == None] = 0
#bad = bad.values
# Число кредитов которые брал клиент
bru = bureau_mean['COUNT_LATE_CREDIT'].values
# Предикт надежности и индексы  
pred = forest.predict_proba(bureau_mean)
ind = pd.Series(bureau_mean.index)
k = pred[::,1]
#Записываем итог

itog = pd.DataFrame({'SK_ID_CURR':ind, 'nadezhnost':k,'count_credit': bru})
itog.to_csv('bureau_itog.csv')
