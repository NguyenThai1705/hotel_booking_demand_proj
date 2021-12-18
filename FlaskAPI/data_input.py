# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:56:51 2021

@author: nguye
"""
import pandas as pd

df_model = pd.read_csv('hotel_booking_cleaned.csv')
df_model = df_model.drop_duplicates()
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

#how to convert new data to OneHotEncoder Data
columns_model = df_model.columns.drop('is_canceled')
newdata = [['Resort Hotel', 342, 'July', 27, 1, 0, 'Breakfast', 'PRT', 'Direct', 'Direct',
            0, 0, 'No Deposit', 0, 'Transient', 0, 0, 'Check-out']]

newdf = pd.DataFrame(newdata, columns=columns_model)

# Apply ohe on newdf
cat_ohe_new = ohe.transform(newdf[categorical_cols])
#Create a Pandas DataFrame of the hot encoded column
ohe_df_new = pd.DataFrame(cat_ohe_new, columns = ohe.get_feature_names(input_features = categorical_cols))
#concat with original data and drop original columns
df_ohe_new = pd.concat([newdf, ohe_df_new], axis=1).drop(columns = categorical_cols, axis=1)

data_in = df_ohe_new.values.tolist()
