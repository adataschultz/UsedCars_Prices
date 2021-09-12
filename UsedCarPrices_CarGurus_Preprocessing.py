# -*- coding: utf-8 -*-
"""
@author: aschu
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uszipcode import SearchEngine

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print('\nUsed Car Prices from CarGurus Preprocessing & EDA')
print('======================================================================')

# Set path
path = r'D:\UsedCarPrices_CarGurus\Data'
os.chdir(path)

# Read data
df = pd.read_csv('used_cars_data.csv', low_memory=False)
df = df.drop_duplicates()
print('\nDimensions of Initial Used Car Data:', df.shape) 
print('======================================================================') 

# Replace empty rows with NA
df = df.replace(r'^\s*$', np.nan, regex=True)

###############################################################################
# Examine Variables for Missingness
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    var_type = df.dtypes
    mis_val_table = pd.concat([mis_val, mis_val_percent, var_type], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Data Type'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print('The selected dataframe has ' + str(df.shape[1]) + ' columns.\n'
          'There are ' + str(mis_val_table_ren_columns.shape[0]) +
          ' columns that have missing values.')
    return mis_val_table_ren_columns

print('\Initial Missing Data Report') 
print(missing_values_table(df))
print('======================================================================') 

###############################################################################
# Drop main_picture_url due to not useful
df = df.drop(['main_picture_url'], axis=1) 

# Remove columns with more than 20% missing
df = df.loc[:, df.isnull().mean() < 0.20]
print('Dimensions after removing variables with > 20% missing data:')
print(df.shape) 
print('======================================================================') 

###############################################################################
# Remove missing rows to retain the most columns
df = df[df.major_options.notna() & df.mileage.notna() & df.engine_displacement.notna() 
         & df.transmission_display.notna() & df.seller_rating.notna()
         & df.engine_cylinders.notna() & df.description.notna() 
         & df.back_legroom.notna() & df.wheel_system.notna() 
         & df.trim_name.notna() & df.interior_color.notna() 
         & df.body_type.notna() & df.exterior_color.notna()  
         & df.franchise_make.notna() & df.torque.notna() 
         & df.highway_fuel_economy.notna() & df.city_fuel_economy & df.power.notna()]

print('Dimensions of data when removing missing rows to retain the most columns:')
print(df.shape)
print('======================================================================') 

###############################################################################
# Examine Year to maintain price comparisons
# Plot year to better visualize
print('Average year in data: ' +  str(round(df['year'].mean(), 2))) 
print('======================================================================') 

# Change path for EDA results
path = r'D:\UsedCarPrices_CarGurus\EDA'
os.chdir(path)

# Count of postings in each year
my_dpi=96 # Change resolution quality
plt.rcParams.update({'font.size': 7})
sns.countplot(x='year', data=df).set_title('Count of Postings in Each Year')
plt.xticks(rotation=90)
plt.savefig('EDA_initialYearCount.png', dpi=my_dpi * 10, bbox_inches='tight')

# Examine distribution of year
sns.boxplot(x=df['year']).set_title('Distribution of Year')
plt.savefig('EDA_initialYearBoxplot.png', dpi=my_dpi * 10, bbox_inches='tight')

# Filter data by year due to count and distribution
df = df.loc[df['year'] >= 2016]
print('\nDimensions of Used Car data after filtering >= 2016:', df.shape) 
print('======================================================================') 

# Count of postings in each ye ar >= 2016
sns.countplot(x='year', data=df).set_title('Count of Postings in Each Year')
plt.savefig('EDA_Year_2016plus_Count.png', dpi=my_dpi * 10, bbox_inches='tight')

###############################################################################
# Examine dealer zip code so price isn't guided by location
# Find length of zipcode as a string
df['dealer_zip_length'] = df['dealer_zip'].str.len()
print('\nCount of different lengths of zipcodes:') 
print(df.dealer_zip_length.value_counts(ascending=False))
print('======================================================================') 

# Filter zipcode to 5 for not full zipcode
df = df[df.dealer_zip_length == 5] # Shortened zipcodes
df = df.drop(['dealer_zip_length'], axis=1)

# Convert to numeric for zipcode search engine
df = df.copy()
df['dealer_zip'] = df['dealer_zip'].astype('int64')

# Find unique zipcodes 
df1 = df.dealer_zip
df1 = df1.unique().tolist()
df1 = pd.DataFrame(df1)
print('\nNumber of unique zipcodes:', df1.shape) 
print('======================================================================') 

# Name columns so it can be used for query
df1.columns = ['dealer_zip_unique']

# Use zipcode to find the state and city corresponding to zipcode
search = SearchEngine(simple_zipcode=False)

# Define function to use zipcode to find state
def zcode_state(x):
    return search.by_zipcode(x).state

df1['State'] = df1['dealer_zip_unique'].fillna(0).astype(int).apply(zcode_state)

# Use zipcode to find city
def zcode_city(x):
    return search.by_zipcode(x).city

df1['City'] = df1['dealer_zip_unique'].fillna(0).astype(int).apply(zcode_city)

# Merge data using right join
df = pd.merge(df, df1, how='right', left_on=['city','dealer_zip'], 
              right_on = ['City','dealer_zip_unique'])
df.drop_duplicates()
df = df.drop(['dealer_zip_unique', 'City'], axis=1)
print('\nMissing data after merging by city and zipcode:') 
print(df.isna().sum()) 
print('======================================================================') 

del df1

# Remove missing from merge due to zipcode not being found in search engine
df = df[df['vin'].notna()]

print('\nDimensions of Used Car data after removing missing from merge:', df.shape)
print('======================================================================') 

# Examine for other variables unable to impute probalistically
def data_type_quality_table(df):
        var_type = df.dtypes
        unique_count = df.nunique()
        mis_val_table = pd.concat([var_type, unique_count], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Data Type', 1 : 'Number Unique'})
        print ('The selected dataframe has ' + str(df.shape[1]) + ' columns.\n')
        return mis_val_table_ren_columns

print('\nData Type & Uniqueness Report') 
print(data_type_quality_table(df))
print('======================================================================') 

###############################################################################
# Drop vars based off missingness, relevancy and similar var types
drop_columns = ['vin', 'latitude', 'longitude',  'trimId', 'trim_name', 
                'sp_id', 'sp_name', 'listing_id', 'model_name', 'description',
                'listed_date', 'make_name']
df.drop(columns=drop_columns, inplace=True)

print('\nDimensions after removing irrelevant variables:', df.shape) 
print('======================================================================') 

###############################################################################
# Process categorical that inches abbreviations to continuous vars
extract_num_from_catVar = lambda series: series.str.split().str[0].astype(np.float)

columns = ['back_legroom', 'wheelbase', 'width', 'length', 'height', 
           'fuel_tank_volume', 'front_legroom', 'maximum_seating']

df[columns] = df[columns].replace({',': '', '--': np.nan}).apply(extract_num_from_catVar)

# Remove missing rows created by transforming previous string var to numeric
df = df[df.back_legroom.notna() & df.front_legroom.notna() 
        & df.fuel_tank_volume.notna() & df.maximum_seating.notna()
        & df.height.notna() & df.length.notna() & df.wheelbase.notna()]

###############################################################################
# Convert variables by splitting string
# Convert torque to torque_rpm
df1 = df.torque.str.replace(',', '').str.split().str[0:4:3]
df1 = pd.DataFrame([[np.nan, np.nan] if type(i).__name__ == 'float' 
                    else np.asarray(i).astype('float') for i in df1])
df1.columns = ['torque_new', 'torque_rpm']

# Concatenate new variables created by string split torque
df = pd.concat([df, df1], axis=1)

# Drop original torque and remove missing from split
df = df.drop(['torque'], axis=1)
df = df[df.torque_rpm.notna()]
df.rename(columns={'torque_new': 'torque'}, inplace=True)

del df1

#Convert power into horsepower_rpm
df1 = df.power.str.replace(',', '').str.split().str[0:4:3]
df1 = pd.DataFrame([[np.nan, np.nan] if type(i).__name__ == 'float' else np.asarray(i).astype('float') for i in df1])
df1.columns = ['horsepower_new', 'horsepower_rpm']

# Concatenate new variables created by string split horsepoqwe
df = pd.concat([df, df1], axis=1)

# Drop original horsepower and remove missing from split
df = df.drop(['horsepower'], axis=1)
df = df[df.horsepower_rpm.notna()]
df.rename(columns={'horsepower_new': 'horsepower'}, inplace=True)

del df1

# Drop vars due to high dimensionality of categorical vars
drop_columns = ['major_options', 'engine_cylinders', 'interior_color',
                'exterior_color', 'engine_type', 'power', 'franchise_dealer',
                'franchise_make', 'transmission_display']
df.drop(columns=drop_columns, inplace=True)

print('\nDimensions after removing cat vars with high dimensionality:', df.shape) 
print('======================================================================') 

###############################################################################
# Write to pickle 
pd.to_pickle(df, "./210911_UsedCars_Preprocessing.pkl")
###############################################################################




