# -*- coding: utf-8 -*-
"""
@author: aschu
"""
import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uszipcode import SearchEngine

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

my_dpi = 96 # Change resolution quality

# Set seed 
seed_value = 42
os.environ['UsedCarPrices_CarGurus_PreprocessingEDA'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

print('\nUsed Car Prices from CarGurus Preprocessing & EDA')
print('======================================================================')

# Set path
path = r'D:\UsedCarPrices_CarGurus\Data'
os.chdir(path)

# Read data
df = pd.read_csv('used_cars_data.csv', low_memory=False)
df = df.drop_duplicates()
print('\nDimensions of initial Used Car Data:', df.shape) 
print('======================================================================') 

# Replace empty rows with NA
df = df.replace(r'^\s*$', np.nan, regex=True)

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

print('\nInitial Missing Data Report') 
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

print('Dimensions when removing missing rows to retain the most columns:')
print(df.shape)
print('======================================================================') 

###############################################################################
# Examine year to maintain price comparisons
print('Average year:' +  str(round(df['year'].mean(), 2))) 
print('======================================================================') 

# Change path for EDA results
path = r'D:\UsedCarPrices_CarGurus\EDA'
os.chdir(path)

# Count of postings in each year
plt.rcParams.update({'font.size': 7})
sns.countplot(x='year', data=df).set_title('Count of Postings in Each Year')
plt.xticks(rotation=90)
plt.savefig('InitialYearCount.png', dpi=my_dpi * 10, bbox_inches='tight')

# Examine distribution of year
sns.boxplot(x=df['year']).set_title('Distribution of Year')
plt.savefig('InitialYearBoxplot.png', dpi=my_dpi * 10, bbox_inches='tight')

# Filter data by year due to count and distribution
df = df.loc[df['year'] >= 2016]
print('\nDimensions of Used Car data after filtering >= 2016:', df.shape) 
print('======================================================================') 

# Count of postings in each year >= 2016
sns.countplot(x='year', data=df).set_title('Count of Postings in Each Year')
plt.savefig('Year_2016plus_Count.png', dpi=my_dpi * 10, bbox_inches='tight')
print('======================================================================') 

###############################################################################
# Examine dealer zip code so price isn't guided by location
# Find length of zipcode as a string
print('\nZipcode processing:')
print('\n')
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
df = df.drop(['dealer_zip_unique', 'City', 'dealer_zip'], axis=1)

print('\nMissing data after merging by city and zipcode:') 
print(df.isna().sum()) 
print('======================================================================') 

del df1

# Remove missing from merge due to zipcode not being found in search engine
df = df[df['vin'].notna()]

print('\nDimensions after removing missing from merge:', df.shape)
print('======================================================================') 

# Examine for other variables unable to impute probalistically
def data_type_quality_table(df):
        var_type = df.dtypes
        unique_count = df.nunique()
        val_table = pd.concat([var_type, unique_count], axis=1)
        val_table_ren_columns = val_table.rename(
        columns = {0 : 'Data Type', 1 : 'Number Unique'})
        print ('The selected dataframe has ' + str(df.shape[1]) + ' columns.\n')
        return val_table_ren_columns

print('\nData Type & Uniqueness Report') 
print(data_type_quality_table(df))
print('======================================================================') 

###############################################################################
# Drop vars based off missingness, relevancy and similar var types
drop_columns = ['vin', 'latitude', 'longitude',  'trimId', 'trim_name', 
                'sp_id', 'sp_name', 'listing_id', 'description']
df.drop(columns=drop_columns, inplace=True)

print('\nDimensions after removing irrelevant variables:', df.shape) 
print('=====================================================================') 

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

print('\nDimensions after removing cat vars with high dimensionality:',
      df.shape) 
print('======================================================================') 

###############################################################################
print('\nExamine when the cars were listed:')
print('\n')

# Convert listed date to monthly
df = df.copy()
df['listed_date'] = pd.to_datetime(df['listed_date'])
df['listed_date_yearMonth'] = df['listed_date'].dt.to_period('M')

print('\nCount of listings in each Year-Month:') 
print(df.listed_date_yearMonth.value_counts(ascending=True)) # Mostly 2020
print('======================================================================') 

# Filter data for highest occurrences: June-September 2020
df = df.loc[(df['listed_date_yearMonth'] >= '2020-06')]

print('\nDimensions after filtering listed_date for highest occurrences: June-September 2020:',
      df.shape) 
print('======================================================================') 

###############################################################################
print('\nExamine where the cars were listed due to differences in standard of living:')
print('\n')

print('\nCount of listings in each US state:') 
print(df.State.value_counts(ascending=True)) # Texas has the most
print('======================================================================') 

# Filter states with the 7 highest counts of listings
df1 = df['State'].value_counts().index[:7]
df = df[df['State'].isin(df1)]

del df1

print('\nDimensions after filtering US states with the 7 highest counts of listings:',
      df.shape) 
print('======================================================================') 

###############################################################################
# Examine dependent variable: price 
print('Summary statistics for price:' + ('\n') +  str(round(df['price'].describe()))) 
print('\n')
print('Median price:' +  str(round(df['price'].median(), 2))) 
print('======================================================================') 

sns.histplot(x=df['price'], kde=True).set_title('Distribution of Price')
plt.savefig('InitialPrice.png', dpi=my_dpi * 10, bbox_inches='tight')

# Filter price less than $50,000
df = df.loc[df['price'] <= 50000.0]
print('\nDimensions after filtering listings <= $50,000:',
      df.shape) 
print('======================================================================') 

sns.histplot(x=df['price'], kde=True).set_title('Distribution of Price Filtered <= $50,000')
plt.savefig('Price_50kOrless.png', dpi=my_dpi * 10, bbox_inches='tight')
print('======================================================================') 

###############################################################################
print('\nExamine categorical variables to see if they should be retained:')
print('\n')
df_car_cat = df.select_dtypes(include = 'object')
df_car_cat = df_car_cat.drop(['city', 'State', 'model_name', 'make_name', 'listing_color'],
                     axis=1)

# Examine select categorical variables with price
plt.rcParams.update({'font.size': 7})
plt.xticks(rotation=90)
fig, ax = plt.subplots(3, 2, figsize=(15, 10))
for var, subplot in zip(df_car_cat, ax.flatten()):
    sns.boxplot(x=var, y='price', data=df, ax=subplot)
plt.tight_layout()  
fig.savefig('QualVar_Boxplot.png', dpi=my_dpi * 10, bbox_inches='tight')

del df_car_cat, fig, ax

#  Examine listing_color and price
plt.rcParams.update({'font.size': 7})
sorted_nb = df.groupby(['listing_color'])['price'].median().sort_values()
sns.boxplot(x=df['listing_color'], y=df['price'], order=list(sorted_nb.index))
plt.xticks(rotation=90)
plt.savefig('Price_listing_color.png', dpi=my_dpi * 10, bbox_inches='tight')

print('\nCount of listings with car color:') 
print(df.listing_color.value_counts(ascending=True)) # Mostly black, UNKNOWN exists
print('======================================================================') 

# Filter observations with the highest number counts for listing color
df1 = df['listing_color'].value_counts().index[:7]
df = df[df['listing_color'].isin(df1)]

del df1

# Remove listings with unknown colors
df = df.loc[df['listing_color'] != 'UNKNOWN']

print('\nDimensions after filtering observations with the highest number counts for listing color:',
      df.shape) 
print('======================================================================') 

# Examine car manufacturer with price
# Find median price of the manufacturer
df1 = df.groupby('make_name')['price'].median().reset_index()
df1.rename(columns={'price': 'make_medianPrice'}, inplace=True)
print('\nMedian price of the manufacturer:') 
print(df1.sort_values('make_medianPrice', ascending=False))
print('======================================================================') 

# Merge data using left join on manufacturer name
df = pd.merge(df, df1, how='left', left_on=['make_name'], 
              right_on = ['make_name'])
df.drop_duplicates()

del df1

# Drop var due to similarity and high dimensionality 
df = df.drop(['city', 'make_name', 'model_name', 'wheel_system'], axis=1)
print('======================================================================') 

###############################################################################
print('\nExamine quantitative variables to see if they should be retained:')
print('\n')
df_num = df.select_dtypes(include = ['float64', 'int64'])

# Histograms
plt.rcParams.update({'font.size': 15})
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); 
plt.savefig('QuantVar_Histograms.png', dpi=my_dpi * 10)

# Correlations
# Find features that are strongly correlated with price
df_num_corr = df_num.corr()['price'][:-19] # Remove price
quant_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There are {} strongly correlated values with Price:\n{}".format(
    len(quant_features_list), quant_features_list))

# Create correlation matrix
df_num = df_num.drop(['price'], axis=1)
corr = df_num.corr(method="spearman")  

# Plot correlation matrix
plt.rcParams.update({'font.size': 2})
sns.heatmap(corr, cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1, 
            annot=False, square=True);
plt.title('Correlation Matrix with Spearman rho')
plt.savefig('CorrelationMatrix_spearman.png', dpi=my_dpi * 10)

# Plot correlation matrix with thresholds
plt.rcParams.update({'font.size': 2})
sns.heatmap(corr[(corr >= 0.8) | (corr <= -0.8)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 4}, square=True);
plt.title('Correlation Matrix with Spearman rho >= 0.8 or <= -0.8')
plt.savefig('CorrelationMatrix_thresholds_spearman.png', dpi=my_dpi * 10)

###############################################################################
# Drop vars not using for modeling
drop_columns = ['seller_rating', 'make_medianPrice', 'listed_date']
df.drop(columns=drop_columns, inplace=True)
print('\nDimensions of data for modeling:', df.shape) 
print('=====================================================================') 

###############################################################################
# Change directory
path = r'D:\UsedCarPrices_CarGurus\Data'
os.chdir(path)

# Write processed data to csv
df.to_csv('usedCars_final.csv', index=False, encoding='utf-8-sig')
###############################################################################