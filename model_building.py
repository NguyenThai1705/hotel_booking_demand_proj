# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 13:37:43 2021

@author: nguye
"""

import pandas as pd
import numpy as np

df= pd.read_csv('hotel_booking_cleaned.csv')
df_model = df.drop_duplicates()
df_model = df_model.dropna().reset_index(drop=True)

#choose relevant columns
irr_col =   ['company','agent','required_car_parking_spaces','booking_changes', 'arrival_date_year',
         'is_repeated_guest','reservation_status_date','stays_in_weekend_nights','stays_in_week_nights',
         'reserved_room_type','assigned_room_type','adults','babies']

df_model.drop(irr_col, axis=1, inplace=True)

#get OneHotEncoder data
# Create a categorical boolean mask
categorical_feature_mask = df_model.dtypes == object
# Filter out the categorical columns into a list for easy reference later on in case you have more than a couple categorical columns
categorical_cols = df_model.columns[categorical_feature_mask].tolist()

# Instantiate the OneHotEncoder Object
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse = False)
# Apply ohe on data
ohe.fit(df_model[categorical_cols])
cat_ohe = ohe.transform(df_model[categorical_cols])

#Create a Pandas DataFrame of the hot encoded column
ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names(input_features = categorical_cols))
#concat with original data and drop original columns
df_ohe = pd.concat([df_model, ohe_df], axis=1).drop(columns = categorical_cols, axis=1)


#train test split
from sklearn.model_selection import train_test_split

X = df_ohe.drop('is_canceled', axis=1)
y = df_ohe.is_canceled.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#multiple logistic regression 
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log = LogisticRegression()
log.fit(X_train, y_train)

log.predict(X_test.iloc[2,:].values.reshape(1, -1))
np.mean(cross_val_score(log, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
print('Logistic Regression Training Accuracy: ', log.score(X_train, y_train))

#saved model
import pickle
pickl = {'model': log}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ))


#test model
file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(X_test.iloc[1,:].values.reshape(1, -1))


#how to convert new data to OneHotEncoder Data
columns_model = df_model.columns.drop('is_canceled')
newdata = [['City Hotel', 19, 'December', 51, 12, 0, 'Half Board', 'FRA', 'Offline TA/TO', 'TA/TO',
            0, 0, 'No Deposit', 0, 'Transient', 108.67, 0, 'Check-out']]

newdf = pd.DataFrame(newdata, columns=columns_model)

# Apply ohe on newdf
cat_ohe_new = ohe.transform(newdf[categorical_cols])
#Create a Pandas DataFrame of the hot encoded column
ohe_df_new = pd.DataFrame(cat_ohe_new, columns = ohe.get_feature_names(input_features = categorical_cols))
#concat with original data and drop original columns
df_ohe_new = pd.concat([newdf, ohe_df_new], axis=1).drop(columns = categorical_cols, axis=1)

# predict on df_ohe_new
predict = model.predict(df_ohe_new)
predict

