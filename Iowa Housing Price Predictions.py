#!/usr/bin/env python
# coding: utf-8

# In[13]:


import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import validation_curve
from sklearn import metrics, svm
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy.stats import boxcox
from math import exp
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict


# In[23]:


# Function created to clean train and test datasets
def clean_dataset(dataset):
    dataset.rename(index=str, columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF","3SsnPorch": "ThreeSsnPorch"},inplace=True)
    
# Fills variables that do not have the specific feature indicated by variable name
# Missing values are not at random
    dataset.Alley.fillna('no_alley', inplace = True)
    dataset.MasVnrType.fillna('None', inplace = True)
    dataset.BsmtQual.fillna('no_basement', inplace = True)
    dataset.BsmtCond.fillna('no_basement', inplace = True)
    dataset.BsmtExposure.fillna('no_basement', inplace = True)
    dataset.BsmtFinType1.fillna('no_basement', inplace = True)
    dataset.BsmtFinType2.fillna('no_basement', inplace = True)
    dataset.FireplaceQu.fillna('no_fireplace', inplace = True)
    dataset.GarageType.fillna('no_garage', inplace = True)
    dataset.GarageFinish.fillna('no_garage', inplace = True)
    dataset.GarageQual.fillna('no_garage', inplace = True)
    dataset.GarageCond.fillna('no_garage', inplace = True)
    dataset.PoolQC.fillna('no_pool', inplace = True)
    dataset.Fence.fillna('no_fence', inplace = True)
    dataset.MiscFeature.fillna('no_misc', inplace = True)
    dataset.MSSubClass.astype('category', inplace=True)

# Imputes zero for numeric variables that have missing values that are not at random
    dataset.BsmtFinSF2.fillna(0, inplace=True)
    dataset.BsmtUnfSF.fillna(0, inplace=True)
    dataset.TotalBsmtSF.fillna(0, inplace=True)
    dataset.BsmtFullBath.fillna(0, inplace=True)
    dataset.BsmtHalfBath.fillna(0, inplace=True)
    dataset.BsmtFinSF1.fillna(0, inplace=True)
    dataset.GarageCars.fillna(0, inplace=True)
    dataset.GarageArea.fillna(0,inplace=True)
    dataset.MasVnrArea.fillna(0 ,inplace = True)

# Imputes most commons column value for cells missing completely at random
    dataset.MSZoning.fillna(dataset.MSZoning.value_counts().idxmax(), inplace=True)
    dataset.Functional.fillna(dataset.Functional.value_counts().idxmax(), inplace=True)
    dataset.SaleType.fillna(dataset.SaleType.value_counts().idxmax(), inplace=True)
    dataset.KitchenQual.fillna(dataset.KitchenQual.value_counts().idxmax(), inplace=True)
    dataset.Utilities.fillna(dataset.Utilities.value_counts().idxmax(), inplace=True)
    dataset.Exterior1st.fillna(dataset.Exterior1st.value_counts().idxmax(), inplace=True)
    dataset.Exterior2nd.fillna(dataset.Exterior2nd.value_counts().idxmax(), inplace=True)
    dataset.Electrical.fillna(dataset.Electrical.value_counts().idxmax(), inplace=True)
    dataset.BsmtFinType2.fillna(dataset.BsmtFinType2.value_counts().idxmax(), inplace=True)
    dataset["LotFrontage"] = dataset.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    
# Creates new variables based on existing ones  
    dataset['GrYrAfter'] = pd.DataFrame(dataset.YearBuilt != dataset.GarageYrBlt)
    dataset['Remodel'] = pd.DataFrame(dataset.YearBuilt != dataset.YearRemodAdd)
    dataset['Remodel_years'] = pd.DataFrame(dataset.YearBuilt - dataset.YearRemodAdd)
    dataset['MSSubClass'] = dataset.MSSubClass.astype(object)
    dataset['after_1980'] = dataset['YearRemodAdd'] > 1985
    dataset['before_1960'] = dataset['YearRemodAdd'] < 1960
    dataset["bn_lf"] = dataset.LotFrontage.apply(lambda x: x > 0)
    dataset["bn_unfbsmt"] = dataset.BsmtUnfSF.apply(lambda x: x > 1)
    dataset["LowQualFinSF2"] = dataset.LowQualFinSF.apply(lambda x: x > 0)
    dataset["PoolArea"] = dataset.PoolArea.apply(lambda x: x > 0)
    dataset["YearBuilt_bin"] = dataset.YearBuilt.apply(lambda x: x < 1950)

#Transforms variables that are skewed
    dataset['LotArea'] = np.log(dataset.LotArea)
    dataset['GrLivArea'] = np.log(dataset.GrLivArea)
    dataset['FirstFlrSF'] = np.log(dataset.FirstFlrSF)
    dataset['LotFrontage'],_ = boxcox(dataset.LotFrontage)
    
    dataset['OverallQual'] = np.log(dataset.OverallQual)
    dataset['OverallCond'] = np.log(dataset.OverallCond)
#     dataset['MasVnrArea'] = np.log(dataset.MasVnrArea)
#     dataset['BsmtFinSF1'] = np.log(dataset.BsmtFinSF1)
#     dataset['BsmtFinSF2'] = np.log(dataset.BsmtFinSF2)
#     dataset['BsmtUnfSF'] = np.log(dataset.BsmtUnfSF)
#     dataset['TotalBsmtSF'] = np.log(dataset.TotalBsmtSF)
#     dataset['TotRmsAbvGrd'] = np.log(dataset.TotRmsAbvGrd)
#     dataset['TotalBsmtSF'] = np.log(dataset.TotalBsmtSF)
#     dataset['GarageCars'] = np.log(dataset.GarageCars)
#     dataset['GarageArea'] = np.log(dataset.GarageArea)
#     dataset['WoodDeckSF'] = np.log(dataset.WoodDeckSF)
#     dataset['KitchenAbvGr'] = np.log(dataset.KitchenAbvGr)
#     dataset['Remodel_years'] = np.log(dataset.KitchenAbvGr)
#     dataset['FullBath'] = np.log(dataset.FullBath)
#     dataset['HalfBath'] = np.log(dataset.HalfBath)
    dataset['MoSold'] = np.log(dataset.MoSold)
#     dataset['LowQualFinSF'] = np.log(dataset.LowQualFinSF)
#     dataset['Fireplaces'] = np.log(dataset.Fireplaces)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    
    dataset.drop('GarageYrBlt', axis=1, inplace=True)

    dataset=pd.get_dummies(dataset, drop_first = True)
    dataset.columns = dataset.columns.str.replace(".", "_")
    dataset.columns = dataset.columns.str.replace("&","_")
    dataset.columns = dataset.columns.str.replace("Exterior", "Ext")
    dataset.rename(index=str, columns={"Ext2nd_Brk Cmn": "Ext_2_BrkCmn","Ext2nd_Wd Sdng":"Ext2nd_WdSdng",
                                     'Ext2nd_Wd Shng':'Ext2nd_WdShng','Ext1st_Wd Sdng':'Ext1st_WdSdng'},inplace=True)
    
    dataset.TotalBsmtSF.fillna(0, inplace=True)
    dataset.Remodel_years.fillna(0, inplace=True)
    
    return dataset
    
    


# In[24]:


train = pd.read_csv('/Users/Ben/Downloads/trainer_kaggle.csv')
train = clean_dataset(train)

# Removes outliers for a more accurate estimate
train = train[train.BsmtFinSF1 < 5000]
train = train[train.GrLivArea < 8.44]
train = train[train.LotFrontage < 40]


train_mismatch = ['RoofMatl_CompShg','Utilities_NoSeWa','Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','HouseStyle_2_5Fin','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Ext1st_ImStucc','Ext1st_Stone','Ext2nd_Other','Heating_GasA','Heating_OthW','Electrical_Mix','GarageQual_Fa','PoolQC_Fa','MiscFeature_TenC']
train.drop(train_mismatch, axis=1, inplace=True)

x = train.loc[:, train.columns != 'SalePrice'].astype(float)
x = pd.DataFrame(preprocessing.scale(x),columns = x.columns)
y = train.SalePrice.astype(float).apply(lambda y: np.log(y))

test = pd.read_csv('/Users/Ben/Downloads/tester_kaggle.csv')
test = clean_dataset(test)

test.drop('MSSubClass_150', axis=1, inplace=True)
test = pd.DataFrame(preprocessing.scale(test),columns = test.columns)

#Creates columns in the train dataset that the test does not have
train_mismatch = ['RoofMatl_CompShg','Utilities_NoSeWa','Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','HouseStyle_2_5Fin','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Ext1st_ImStucc','Ext1st_Stone','Ext2nd_Other','Heating_GasA','Heating_OthW','Electrical_Mix','GarageQual_Fa','PoolQC_Fa','MiscFeature_TenC']
for col in train_mismatch:
    test[col] = 0


# In[20]:


warnings.filterwarnings('ignore')

#Fitting models on the training dataset and evaluating the MSE using cross validation
eNet = ElasticNet(fit_intercept=True, alpha = .1, l1_ratio = .001)
ridge = Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, solver='svd', tol = 1e-05)
lasso = Lasso(alpha=.001, copy_X=True, fit_intercept=True, max_iter=25,normalize=False,positive=False,selection='cyclic')
gbr = GradientBoostingRegressor(learning_rate = .1, n_estimators = 400,subsample = 0.7,
                               max_depth = 2, min_samples_leaf= 2, min_samples_split = 3)
rf = RandomForestRegressor(bootstrap = True,max_depth = 30, min_samples_leaf = 2, min_samples_split = 3,
                           n_estimators = 100, oob_score = True, warm_start = True)
xgb = XGBRegressor(n_estimators = 400,gamma = 0,max_depth = 3,subsample = .8, reg_lambda = 1, alpha = 0)
svr = SVR(epsilon = 0,gamma = .000001, C = 1000, shrinking = True)


models = [eNet,ridge,lasso,gbr,rf,xgb,svr]
models = [eNet,ridge]

for model in models:
    model.fit(x,y)
    score =  cross_val_score(model, x, y, cv=10,scoring = 'neg_mean_squared_error').mean()
    print((score*-1)**.5)
    


# In[129]:


# Template for submitting predictions to Kaggle

model = lasso
model.fit(x,y)
model_pred = pd.Series(model.predict(test))
subm = pd.read_csv('/Users/Ben/Downloads/sample_submission.csv')
subm.drop('SalePrice', axis=1, inplace=True)
subm['SalePrice'] = model_pred.apply(lambda price: exp(price))
subm
subm.to_csv('ggbbbrrrr.csv',index=False)

