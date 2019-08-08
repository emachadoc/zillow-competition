'''Import libraries'''

%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import zipfile
import re
from scipy import stats
from catboost import CatBoostRegressor
from tqdm import tqdm
import gc
import datetime as dt

'''Once we have imported the necessary libraries, we are going to define
our data path and then, load the data:'''

data_path = './data/'
print('Loading Properties ...')
properties2016 = pd.read_csv(data_path + 'properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv(data_path + 'properties_2017.csv', low_memory = False)

properties2016.shape
properties2017.shape

'''We are going to specify the parameter 'parse_dates' to make the column
'transactiondate' in train files date formatted:'''

print('Loading Train ...')
train2016 = pd.read_csv(data_path + 'train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = pd.read_csv(data_path + 'train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

train2016.shape
train2017.shape

'''Now we are going to parse dates to get one column for year, one for month,
one for day and one for quarter:'''

def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 + df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df

train2016 = add_date_features(train2016)
train2017 = add_date_features(train2017)

'''Our sample submission will be a file with the parcel id for each property and
six prediction columns for different dates (last 2016 quarter and last 2017 quarter):'''

print('Loading Sample ...')
sample_submission = pd.read_csv(data_path + 'sample_submission.csv', low_memory = False)

'''As we are going to apply the first model, cat boost, to the column fo
 201610 predictions, we remove the other columns:'''

 sample_submission_model_1 = sample_submission.drop(columns = ['201611', '201612', '201710', '201711', '201712'])

 '''We will merge properties and train files to get all data grouped:'''

print('Merging Train with Properties ...')
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')

train2016.shape
train2017.shape

'''As we should not use the 2016 tax values when predicting log errors against the
2016 log errors, we will omit this information for the 2016 predicitons columns:'''

print('Tax Features 2017  ...')
train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan
train2017.iloc[:, train2017.columns.str.endswith('taxvaluedollarcnt')] = np.nan

'''Now we will generate our training and test files to run the model, the training
file will be all properties and train from 2016 and 2017, while for the test, we
will merge the sample submission with the unique properties file:'''

print('Concat Train 2016 & 2017 ...')
train_df = pd.concat([train2016, train2017], axis = 0)
test_df = pd.merge(sample_submission_model_1[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')

train_df.shape
test_df.shape

'''To optimize memory management, we are using garbage collections algorithms (GC),
which solve reference cycles (when one or more objects are referencing each other)
that reference counting cannot detect:'''

del properties2016, properties2017, train2016, train2017
gc.collect();

'''Let's do some feature engineering. We have to deal with missing values, where
we will establish a threshold of 98% missing values to remove those fields, and
also with features with one unique value and others that we do not want to use in
our training:'''

print('Missing data fields to remove ...')
missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print(exclude_missing)
print("We exclude: %s" % len(exclude_missing))

print ("Remove features with one unique value ...")
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print(exclude_unique)
print("We exclude: %s" % len(exclude_unique))

del num_rows, missing_perc_thresh
gc.collect();

print ("Define training features ...")
exclude_other = ['parcelid', 'logerror', 'propertyzoningdesc']
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other:
        train_features.append(c)
print(train_features)
print("We use these for training: %s" % len(train_features))

'''Now we have to deal with categorical features:'''

print ("Define categorical features ...")
cat_feature_inds = []
cat_unique_thresh = 100
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'number' in c:
        cat_feature_inds.append(i)
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

'''As we are going to use a tree based model, we will replace NaN values by '-999'
so they do not interfere with proper data:'''

print ("Replacing NaN values by -999 ...")
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

'''Training start:'''

print ("Training model 1: Cat boost ...")
X_train = train_df[train_features]
y_train = train_df.logerror
print(X_train.shape, y_train.shape)

test_df['transactiondate'] = pd.Timestamp('2016-12-01') 
test_df = add_date_features(test_df)
X_test = test_df[train_features]
print(X_test.shape)

y_pred = 0.0
model = CatBoostRegressor(
    iterations = 630, learning_rate = 0.02,
    depth = 6, l2_leaf_reg = 3,
    loss_function = 'MAE',
    eval_metric = 'MAE',
    random_seed = i)
model.fit(
    X_train, y_train,
    cat_features = cat_feature_inds)
y_pred += model.predict(X_test)

submission = pd.DataFrame({
    'ParcelId': test_df['ParcelId'],
})
test_dates = {
    '201610': pd.Timestamp('2016-09-30')
}
for label, test_date in test_dates.items():
    print("Predicting for: %s ... " % (label))
    submission[label] = y_pred
    
submission.to_csv(data_path + 'Model_1_CatBoost.csv', float_format='%.6f',index=False)

predictions_model_1 = pd.read_csv(data_path + 'Model_1_CatBoost.csv')
predictions_model_1.head()



