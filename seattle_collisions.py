
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import seaborn as sns
from sklearn.cluster import KMeans
os.chdir(r"C:\Users\jason\Documents\GitHub\Coursera_Capstone_IBM-Data-Science-Professional-Certificate")

# Read file
df = pd.read_csv("Data-Collisions.csv", encoding='utf8')
data = df.copy()
del df

data.head()

data.columns
data.shape
data.info()
data.dtypes

col_na = round(100*(data.isna().sum()/len(data)), 2)
col_nunique = data.nunique()    
df_summary = pd.DataFrame({"data_type": data.dtypes,
                           "percent_missing_values": col_na,
                           "total_unique_values": col_nunique}).sort_values(by=["percent_missing_values"],
                           ascending=False)
df_summary.head(25)
del col_na, col_nunique
df_summary

data['SEVERITYCODE'].unique()
data['SEVERITYCODE'] = data['SEVERITYCODE'].replace(1,0)
data['SEVERITYCODE'] = data['SEVERITYCODE'].replace(2,1)

data['INCDATE'] = data['INCDATE'].str.slice(stop=10)
data['INCDATE'] = data['INCDATE'].astype('datetime64[ns]')
data['year'] = data['INCDATE'].dt.year
data['month'] = data['INCDATE'].dt.month
data['day'] = data['INCDATE'].dt.day
data['weekday'] = data['INCDATE'].dt.weekday


data['INCDTTM'].unique()
data['AMPM'] = data['INCDTTM'].str.slice(start=-2)
data['AMPM'] = np.where((data['AMPM'] == 'AM') | 
                        (data['AMPM'] == 'PM'), 
                        data['AMPM'], np.nan)

data['TIME'] = data['INCDTTM'].str.slice(start=-11)
data['HOUR'] = data['TIME'].str.slice(stop=2)
data['HOUR'] = data['HOUR'].str.replace('/','')
data['HOUR'] = data['HOUR'].str.replace(' ','')
data['HOUR'].unique()
data['HOUR'] = data['HOUR'].astype(int)
data['HOUR'] = np.where(data['AMPM'] == 'PM', data['HOUR']+12, data['HOUR'])


data.columns
sns.countplot(y="SEVERITYCODE", hue="ADDRTYPE", data=data)
data[['SEVERITYCODE']].groupby(data['ADDRTYPE']).mean().sort_values(by=["SEVERITYCODE"], ascending=False)

sns.countplot(y="month", hue="SEVERITYCODE", data=data)
data[['SEVERITYCODE']].groupby(data['month']).mean().sort_values(by=["SEVERITYCODE"], ascending=False)

data[['SEVERITYCODE']].groupby(data['ST_COLCODE']).mean().sort_values(by=["SEVERITYCODE"], ascending=False)

data[['SEVERITYCODE']].groupby(data['COLLISIONTYPE']).mean().sort_values(by=["SEVERITYCODE"], ascending=False)


features = ['ST_COLCODE','COLLISIONTYPE','SDOT_COLCODE','SEGLANEKEY','CROSSWALKKEY','PEDCOUNT',
            'PEDCYLCOUNT']
for i in features:
    fig, ax = plt.subplots(figsize=(16,9))
    hrvsday = data.pivot_table(values='SEVERITYCODE',index='HOUR',columns=i,aggfunc='mean')
    ax.set_title(i + " vs. Hour in average")
    sns.heatmap(hrvsday,cmap='magma_r', ax=ax) #  Monday=0, Sunday=6
#    plt.savefig("d_higher " + str(i) + " vs. Hour in average"+'.jpg', transparent=True)
    plt.show()



data.dtypes
data['ST_COLCODE'].unique()
data['ST_COLCODE'] = data['ST_COLCODE'].astype("category")
data['ST_COLCODE'] = data['ST_COLCODE'].astype("str")
data['ST_COLCODE'] = data['ST_COLCODE'].replace(' ',np.nan)
data['ST_COLCODE'] = data['ST_COLCODE'].astype(float)

data.columns
data['High_risk_ST_COLCODE'] = np.where((data['ST_COLCODE'] == 0) | 
                            (data['ST_COLCODE'] == 1) |
                            (data['ST_COLCODE'] == 2) |
                            (data['ST_COLCODE'] == 3) |
                            (data['ST_COLCODE'] == 4) |
                            (data['ST_COLCODE'] == 5) |
                            (data['ST_COLCODE'] == 6) |
                            (data['ST_COLCODE'] == 7) |
                            (data['ST_COLCODE'] == 8) |
                            (data['ST_COLCODE'] == 24) |
                            (data['ST_COLCODE'] == 45) |
                            (data['ST_COLCODE'] == 49) |
                            (data['ST_COLCODE'] == 52) |
                            (data['ST_COLCODE'] == 53) |
                            (data['ST_COLCODE'] == 87) |
                            (data['ST_COLCODE'] == 10) |
                            (data['ST_COLCODE'] == 13) |
                            (data['ST_COLCODE'] == 14) |
                            (data['ST_COLCODE'] == 21) |
                            (data['ST_COLCODE'] == 25) |
                            (data['ST_COLCODE'] == 28) |
                            (data['ST_COLCODE'] == 30) |
                            (data['ST_COLCODE'] == 66) |
                            (data['ST_COLCODE'] == 73) |
                            (data['ST_COLCODE'] == 74) |
                            (data['ST_COLCODE'] == 84) |
                            (data['ST_COLCODE'] == 88)
                            
                        ,1, 0)

data.columns
data['High_risk_COLLISIONTYPE'] = np.where((data['COLLISIONTYPE'] == 'Pedestrian') | 
                            (data['COLLISIONTYPE'] == 'Cycles') |
                            (data['COLLISIONTYPE'] == 'Head On') |
                            (data['COLLISIONTYPE'] == 'Rear Ended')                            
                            ,1, 0)

data.columns
data['High_risk_SDOT_COLCODE'] = np.where((data['SDOT_COLCODE'] == 44) | 
                            (data['SDOT_COLCODE'] == 58) |
                            (data['SDOT_COLCODE'] == 61) |
                            (data['SDOT_COLCODE'] == 69) |
                            (data['SDOT_COLCODE'] == 64) |
                            (data['SDOT_COLCODE'] == 66) |
                            (data['SDOT_COLCODE'] == 22) |
                            (data['SDOT_COLCODE'] == 24) |
                            (data['SDOT_COLCODE'] == 56) |
                            (data['SDOT_COLCODE'] == 51) |
                            (data['SDOT_COLCODE'] == 18) |
                            (data['SDOT_COLCODE'] == 55) |
                            (data['SDOT_COLCODE'] == 53) |
                            (data['SDOT_COLCODE'] == 21) |
                            (data['SDOT_COLCODE'] == 54) |
                            (data['SDOT_COLCODE'] == 29) |
                            (data['SDOT_COLCODE'] == 23) |
                            (data['SDOT_COLCODE'] == 68) |
                            (data['SDOT_COLCODE'] == 52) 
                            ,1, 0)

data['High_risk_SEGLANEKEY'] = np.where((data['SEGLANEKEY'] > 0)
                            ,1, 0)

data['High_risk_CROSSWALKKEY'] = np.where((data['CROSSWALKKEY'] > 0)
                            ,1, 0)

data['High_risk_PEDCOUNT'] = np.where((data['PEDCOUNT'] > 0)
                            ,1, 0)

data['High_risk_PEDCYLCOUNT'] = np.where((data['PEDCYLCOUNT'] > 0)
                            ,1, 0)


data = data.drop(columns=['OBJECTID','INCKEY','COLDETKEY','SDOTCOLNUM', 'INCDATE',
       'INCDTTM', 'REPORTNO','COLDETKEY', 'INCKEY', 'OBJECTID','EXCEPTRSNDESC',
       'ST_COLDESC','SDOT_COLDESC','SEVERITYDESC','SEVERITYCODE.1',
       'EXCEPTRSNDESC', 'INATTENTIONIND', 'INTKEY', 'EXCEPTRSNCODE',
       'SDOTCOLNUM','TIME','SDOT_COLCODE','ST_COLCODE','PEDROWNOTGRNT',
       'PERSONCOUNT','PEDCOUNT','COLLISIONTYPE','STATUS','year','JUNCTIONTYPE',
       'HITPARKEDCAR','PEDCYLCOUNT','VEHCOUNT','AMPM',
       'SEGLANEKEY', 'CROSSWALKKEY', 'SPEEDING'])

col = ['month', 'day', 'ADDRTYPE', 'UNDERINFL', 'HOUR']

for col in col:
    data[col] = data[col].astype("category")

data.info()

col_na = round(100*(data.isna().sum()/len(data)), 2)
col_nunique = data.nunique()
df_summary = pd.DataFrame({"data_type": data.dtypes,
                           "percent_missing_values": col_na,
                           "total_unique_values": col_nunique}).sort_values(by=["percent_missing_values"],
                           ascending=False)
df_summary.head(25)
del col_na, col_nunique
df_summary




data = data[['SEVERITYCODE', 'X', 'Y','ADDRTYPE','High_risk_ST_COLCODE',
             'High_risk_COLLISIONTYPE','HOUR','High_risk_SDOT_COLCODE',
             'High_risk_SEGLANEKEY','High_risk_CROSSWALKKEY','High_risk_PEDCOUNT','High_risk_PEDCYLCOUNT']]

data = pd.get_dummies(data, columns=['ADDRTYPE','High_risk_ST_COLCODE',
                                     'High_risk_COLLISIONTYPE','HOUR','PEDROWNOTGRNT',
                                     'High_risk_SDOT_COLCODE','High_risk_SEGLANEKEY',
                                     'High_risk_CROSSWALKKEY','High_risk_PEDCOUNT','High_risk_PEDCYLCOUNT'])


data = data.dropna(how='any')


# Split train test
X = data.drop(columns=['SEVERITYCODE'])
y = data['SEVERITYCODE']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# over sampling train data
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)
X_train = pd.DataFrame(data=X_train)
X_train.columns = list(X_test)

import os
import random
random.seed (2)


""" Lightgbm classifier """
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

lgb_train = lgbm.Dataset(X_train, y_train)
lgb_eval = lgbm.Dataset(X_test, y_test, reference=lgb_train)


# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 2,
    'learning_rate': 0.001,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.3,
    'bagging_freq': 2,
    'verbose': 0,
    'max_depth': 10,
    'n_jobs': 4,
}


# train
gbm = lgbm.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50)


clf_train_pred = gbm.predict(X_train)
clf_train_pred = np.where(clf_train_pred > 0.5, 1, 0)
clf_test_pred = gbm.predict(X_test)
clf_test_pred = np.where(clf_test_pred > 0.5, 1, 0)

lgbm.plot_importance(gbm, max_num_features=20)


""" Evaluation """
from sklearn.metrics import roc_auc_score, roc_curve, auc,mean_squared_error\
                            ,mean_absolute_error, confusion_matrix\
                            ,classification_report, accuracy_score, precision_score
                            
#Confusion matrix
cm = confusion_matrix(y_test,clf_test_pred)
print("Test_Confusion_matrix: \n",cm)

#cmap = sns.cubehelix_palette(light=1, as_cmap=True)
#res = sns.heatmap(cm, annot=True, fmt='.0f', cmap=plt.cm.Blues)
#plt.show()

print("Train_roc_auc_score: ",roc_auc_score(y_train, clf_train_pred))

print("Test_roc_auc_score: ",roc_auc_score(y_test, clf_test_pred))


#Sensitivity
total=sum(sum(cm))
sensitivity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Test_Sensitivity : ', sensitivity1 )

#Specificity
specificity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Test_Specificity : ', specificity1)

#Accuracy
print("Test_accuracy_score:",accuracy_score(y_test, clf_test_pred))

print(classification_report(y_test, clf_test_pred))

import os
import random
random.seed (2)

""" Naive_bayes """
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
clf = GaussianNB()
clf.fit(X_train, y_train)
clf_train_pred = clf.predict(X_train)
clf_test_pred = clf.predict(X_test)


import eli5 
from eli5.sklearn import PermutationImportance
from IPython.display import display
perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test) 
html_obj = eli5.show_weights(perm, feature_names = X_test.columns.tolist())
with open(r'C:\Users\jason\Desktop\importance.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))


""" Evaluation """
from sklearn.metrics import roc_auc_score, roc_curve, auc,mean_squared_error\
                            ,mean_absolute_error, confusion_matrix\
                            ,classification_report, accuracy_score, precision_score
                            
#Confusion matrix
cm = confusion_matrix(y_test,clf_test_pred)
print("Test_Confusion_matrix: \n",cm)

#cmap = sns.cubehelix_palette(light=1, as_cmap=True)
#res = sns.heatmap(cm, annot=True, fmt='.0f', cmap=plt.cm.Blues)
#plt.show()

print("Train_roc_auc_score: ",roc_auc_score(y_train, clf_train_pred))

print("Test_roc_auc_score: ",roc_auc_score(y_test, clf_test_pred))


#Sensitivity
total=sum(sum(cm))
sensitivity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Test_Sensitivity : ', sensitivity1 )

#Specificity
specificity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Test_Specificity : ', specificity1)

#Accuracy
print("Test_accuracy_score:",accuracy_score(y_test, clf_test_pred))

print(classification_report(y_test, clf_test_pred))

import os
import random
random.seed (2)

""" Xgboost classifier """
import xgboost as xgb

clf = xgb.XGBClassifier(max_depth=7, learning_rate=0.001, n_estimators=5000, 
                        verbosity=1, objective='binary:logistic', 
                        booster='gbtree', n_jobs=4, gamma=0, 
                        min_child_weight=1, max_delta_step=0, subsample=1, 
                        colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, 
                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
                        random_state=0)
clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
        verbose=False) # Change verbose to True if you want to see it train

plt.figsize=(16, 9)
xgb.plot_importance(clf, height=0.5, max_num_features=25)
#plt.savefig('feature importance.jpg', transparent=True)

clf_train_pred = clf.predict(X_train)
clf_test_pred = clf.predict(X_test)



""" Evaluation """
from sklearn.metrics import roc_auc_score, roc_curve, auc,mean_squared_error\
                            ,mean_absolute_error, confusion_matrix\
                            ,classification_report, accuracy_score, precision_score
                            
#Confusion matrix
cm = confusion_matrix(y_test,clf_test_pred)
print("Test_Confusion_matrix: \n",cm)

#cmap = sns.cubehelix_palette(light=1, as_cmap=True)
#res = sns.heatmap(cm, annot=True, fmt='.0f', cmap=plt.cm.Blues)
#plt.show()

print("Train_roc_auc_score: ",roc_auc_score(y_train, clf_train_pred))

print("Test_roc_auc_score: ",roc_auc_score(y_test, clf_test_pred))


#Sensitivity
total=sum(sum(cm))
sensitivity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Test_Sensitivity : ', sensitivity1 )

#Specificity
specificity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Test_Specificity : ', specificity1)

#Accuracy
print("Test_accuracy_score:",accuracy_score(y_test, clf_test_pred))

print(classification_report(y_test, clf_test_pred))

import os
import random
random.seed (2)

""" MLPClassifier """
from sklearn.neural_network import MLPClassifier


clf = MLPClassifier(hidden_layer_sizes=(15,10,5,3), activation='logistic',
                    solver='adam', alpha=0.001, batch_size='auto',
                    learning_rate='invscaling', learning_rate_init=0.0001,
                    power_t=0.5, max_iter=200, shuffle=True, random_state=None,
                    tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                    nesterovs_momentum=True, early_stopping=False,
                    validation_fraction=0.1, beta_1=0.1, beta_2=0.9999,
                    epsilon=1e-08, n_iter_no_change=10)
clf.fit(X_train, y_train)
clf_train_pred = clf.predict(X_train)
clf_test_pred = clf.predict(X_test)


""" Evaluation """
from sklearn.metrics import roc_auc_score, roc_curve, auc,mean_squared_error\
                            ,mean_absolute_error, confusion_matrix\
                            ,classification_report, accuracy_score, precision_score
                            
#Confusion matrix
cm = confusion_matrix(y_test,clf_test_pred)
print("Test_Confusion_matrix: \n",cm)

#cmap = sns.cubehelix_palette(light=1, as_cmap=True)
#res = sns.heatmap(cm, annot=True, fmt='.0f', cmap=plt.cm.Blues)
#plt.show()

print("Train_roc_auc_score: ",roc_auc_score(y_train, clf_train_pred))

print("Test_roc_auc_score: ",roc_auc_score(y_test, clf_test_pred))


#Sensitivity
total=sum(sum(cm))
sensitivity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Test_Sensitivity : ', sensitivity1 )

#Specificity
specificity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Test_Specificity : ', specificity1)

#Accuracy
print("Test_accuracy_score:",accuracy_score(y_test, clf_test_pred))

print(classification_report(y_test, clf_test_pred))