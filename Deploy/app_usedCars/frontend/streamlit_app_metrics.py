import streamlit as st
st.set_page_config(page_title='Predicting the Price of Used Vehicles',
                   layout='wide')
import os
import random
import numpy as np
import warnings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import dill as pickle
import lightgbm
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
warnings.filterwarnings('ignore')

seed_value = 42
os.environ['usedCars_GPU'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Set path
#path = '/mnt/UsedCars_Prices/Deploy/app_usedCars/frontend'
path = '/mount/src/usedcars_prices/Deploy/app_usedCars/frontend'
os.chdir(path)

st.markdown("<h1 style='text-align: center; color: black;'>Predicting the Price of Used Vehicles from CarGurus</h1>", unsafe_allow_html=True)

# Set number of columns
col1, col2 , col3 = st.columns(3)

# Price State: train/test
path_to_html = path + '/static/traintest_priceState.html'

def plot_traintest_monthPriceState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        traintest_priceState = f.read()
    with col2:
        st.components.v1.html(traintest_priceState, height=500, width=700)
    return path_to_html

plot_traintest_monthPriceState(path_to_html)

# Year State: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = path + '/static/train_yearState.html'

def plot_train_yearState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_yearState = f.read()
    with col1:
        st.components.v1.html(train_yearState, height=500, width=700)
    return path_to_html

plot_train_yearState(path_to_html)

# Year State: Test
path_to_html = path + '/static/test_yearState.html'

def plot_test_yearState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_yearState = f.read()
    with col2:
        st.components.v1.html(test_yearState, height=500, width=700)
    return path_to_html

plot_test_yearState(path_to_html)

# Month Price State: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = path + '/static/train_monthPriceState.html'

def plot_train_monthPriceState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_monthPriceState = f.read()
    with col1:
        st.components.v1.html(train_monthPriceState, height=500, width=700)
    return path_to_html

plot_train_monthPriceState(path_to_html)

# Month Price State: Test
path_to_html = path + '/static/test_monthPriceState.html'

def plot_test_monthPriceState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_monthPriceState = f.read()
    with col2:
        st.components.v1.html(test_monthPriceState, height=500, width=700)
    return path_to_html

plot_test_monthPriceState(path_to_html)

# Price Color: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = path + '/static/train_priceColor.html'

def plot_train_priceColor(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_priceColor = f.read()
    with col1:
        st.components.v1.html(train_priceColor, height=500, width=700)
    return path_to_html

plot_train_priceColor(path_to_html)

# Price Color: Test
path_to_html = path + '/static/test_priceColor.html'

def plot_test_priceColor(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_priceColor = f.read()
    with col2:
        st.components.v1.html(test_priceColor, height=500, width=700)
    return path_to_html

plot_test_priceColor(path_to_html)

# Price Color State: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = path + '/static/train_priceColorState.html'

def plot_train_PriceColorState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_PriceColorState = f.read()
    with col1:
        st.components.v1.html(train_PriceColorState, height=500, width=700)
    return path_to_html

plot_train_PriceColorState(path_to_html)

# Price Color State: Test
path_to_html = path + '/static/test_priceColorState.html'

def plot_test_PriceColorState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_PriceColorState = f.read()
    with col2:
        st.components.v1.html(test_PriceColorState, height=500, width=700)
    return path_to_html

plot_test_PriceColorState(path_to_html)

# Days on Market Color State: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = path + '/static/train_domColorState.html'

def plot_train_domColorState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_domColorState = f.read()
    with col1:
        st.components.v1.html(train_domColorState, height=500, width=700)
    return path_to_html

plot_train_domColorState(path_to_html)

# Days on Market Color State: Test
path_to_html = path + '/static/test_domColorState.html'

def plot_test_domColorState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_domColorState = f.read()
    with col2:
        st.components.v1.html(test_domColorState, height=500, width=700)
    return path_to_html

plot_test_domColorState(path_to_html)

###################################################################################################################
###################################################################################################################
st.subheader('Data Monitoring', divider='blue')

# Data Quality
path_to_html_quality = path + '/static/DataQualityPreset_report.html' 

path_to_html_presets = path + '/static/data_qualityTestPresets_report.html' 

col1, col2 = st.columns(2)

def plot_dataQuality(path_to_html_quality): 
    with open(path_to_html_quality,'r') as f: 
        data_quality = f.read()

    with col1:
        st.subheader('Data Quality')
        st.components.v1.html(data_quality, scrolling=True, height=900, width=900)
    return data_quality

plot_dataQuality(path_to_html_quality)

def plot_dataQualityPresets(path_to_html_presets):
    
    with open(path_to_html_presets,'r') as f: 
        data_quality_presets = f.read()

    with col2:
        st.components.v1.html(data_quality_presets, scrolling=True, height=900, width=900)
    return data_quality_presets

plot_dataQualityPresets(path_to_html_presets)

###################################################################################################################
# Data Integrity and Stability
path_to_html_integrity = path + '/static/data_integrity_dataset_report_SummaryMissing.html' 

path_to_html_stability = path + '/static/data_stabilityTestPresets_report.html'

col1, col2 = st.columns(2)

def plot_dataIntegrity(path_to_html_integrity):
    
    with open(path_to_html_integrity,'r') as f: 
        data_integrity = f.read()

    with col1:
        st.subheader('Data Integrity')
        st.components.v1.html(data_integrity, scrolling=True, height=900, width=900)
    
    return data_integrity
    
plot_dataIntegrity(path_to_html_integrity)

def plot_dataStabilty(path_to_html_stability):
    
    with open(path_to_html_stability,'r') as f: 
        data_stability = f.read()
    with col2:
        st.subheader('Data Stability')
        st.components.v1.html(data_stability, scrolling=True, height=900, width=900)
    
    return data_stability

plot_dataStabilty(path_to_html_stability)

###################################################################################################################
# Data Drift
path_to_html_drift = path + '/static/data_drift_report.html'

col1, col2 , col3 = st.columns(3)

def plot_dataDrift(path_to_html_drift):

    with open(path_to_html_drift,'r') as f: 
        data_drift = f.read()
    with col2:
        st.subheader('Data Drift')
        st.components.v1.html(data_drift, scrolling=True, height=900, width=900) 
    return data_drift

data_drift = plot_dataDrift(path_to_html_drift)  

###################################################################################################################
###################################################################################################################
# Load data
@st.cache_data
def load_data():
    try:
        train_data = pd.read_parquet(path + '/data/usedCars_trainSet.parquet.gzip')
        test_data = pd.read_parquet(path + '/data/usedCars_testSet.parquet.gzip')
        return train_data, test_data
    except Exception as ex:
        raise(f'Error in loading file: {ex}', str(ex))
        
# Load models
os.environ['LGB_MODEL_DIR'] = path + '/lightgbm/model/usedcars_lgbm_model.pkl'
os.environ['CAT_MODEL_DIR'] = path + '/catboost/model/usedcars_cat_model'
os.environ['XGB_MODEL_DIR'] = path + '/xgboost/model/usedcars_xgb_model.bin'

@st.cache_resource
def load_lgb_model():
    lgb_path = os.environ['LGB_MODEL_DIR']
    model = joblib.load(open(lgb_path,'rb'))
    return model

@st.cache_resource
def load_cat_model():
    cat_path = os.environ['CAT_MODEL_DIR']
    model = CatBoostRegressor()
    model.load_model(cat_path)
    return model

@st.cache_resource
def load_xgb_model():
    xgb_path = os.environ['XGB_MODEL_DIR']
    model = xgb.Booster()
    model.load_model(xgb_path)
    return model

# Load data
trainDF, testDF = load_data()

train_label = trainDF[['price']]
test_label = testDF[['price']]

train_features = trainDF.drop(columns = ['price'])
test_features = testDF.drop(columns = ['price'])

train_Features = pd.get_dummies(train_features, drop_first=True)
test_Features = pd.get_dummies(test_features, drop_first=True)
    
categorical_features_indices = ['body_type', 'fuel_type', 'listing_color',
                                'transmission', 'wheel_system_display', 'State',
                                'listed_date_yearMonth', 'is_new']

dtrain = xgb.DMatrix(train_Features, label=train_label)
dtest = xgb.DMatrix(test_Features, label=test_label)

model_lgb = load_lgb_model()
model_cat = load_cat_model()
model_xgb = load_xgb_model()

###################################################################################################################
st.subheader('Models Metrics for Used Vehicle Price Prediction Using LightGBM', divider='blue')

y_train_pred = model_lgb.predict(train_Features)
y_test_pred = model_lgb.predict(test_Features)

st.write('MAE train: %.3f, test: %.3f' % (
        mean_absolute_error(train_label, y_train_pred),
        mean_absolute_error(test_label, y_test_pred)))
st.write('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_label, y_train_pred),
        mean_squared_error(test_label, y_test_pred)))
st.write('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_label, y_train_pred, squared=False),
        mean_squared_error(test_label, y_test_pred, squared=False)))
st.write('R^2 train: %.3f, test: %.3f' % (
        r2_score(train_label, y_train_pred),
        r2_score(test_label, y_test_pred)))    

st.subheader('LightGBM: Feature Importance', divider='blue')

st.image(path + '/lightgbm/results/LGBM_FeatureImportance.png')

# SHAP
st.subheader('LightGBM: Model-based SHAP', divider='blue')

col1, col2 = st.columns(2)

# Train set
path_train_summary = path + '/lightgbm/results/LGBM_ShapSummary_TrainSet.png'

path_train_force = path + '/lightgbm/results/LGBM_ShapForce_TrainSet.png'

def plot_lgb_train_shap(path_train_summary, path_train_force):

    with col1:
        summary = st.image(path_train_summary)
    with col2:
        force = st.image(path_train_force)
    return summary, force

plot_lgb_train_shap(path_train_summary, path_train_force)  

# Test set
col1, col2 = st.columns(2)

path_test_summary = path + '/lightgbm/results/LGBM_ShapSummary_TestSet.png'

path_test_force = path + '/lightgbm/results/LGBM_ShapForce_TestSet.png'

def plot_lgb_test_shap(path_test_summary, path_test_force):

    with col1:
        summary = st.image(path_test_summary)
    with col2:
        force = st.image(path_test_force)
    return summary, force

plot_lgb_test_shap(path_test_summary, path_test_force)

###################################################################################################################
st.subheader('Models Metrics for Used Vehicle Price Prediction Using Catboost', divider='blue')

y_train_pred = model_cat.predict(train_features)
y_test_pred = model_cat.predict(test_features)

st.write('MAE train: %.3f, test: %.3f' % (
        mean_absolute_error(train_label, y_train_pred),
        mean_absolute_error(test_label, y_test_pred)))
st.write('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_label, y_train_pred),
        mean_squared_error(test_label, y_test_pred)))
st.write('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_label, y_train_pred, squared=False),
        mean_squared_error(test_label, y_test_pred, squared=False)))
st.write('R^2 train: %.3f, test: %.3f' % (
        r2_score(train_label, y_train_pred),
        r2_score(test_label, y_test_pred)))    

st.subheader('Catboost: Feature Importance', divider='blue')

st.image(path + '/catboost/results/Cat_FeatureImportance.png')

# SHAP
st.subheader('Catboost: Model-based SHAP', divider='blue')

col1, col2 = st.columns(2)

# Train set
path_train_summary = path + '/catboost/results/Cat_ShapSummary_TrainSet.png'

path_train_force = path + '/catboost/results/Cat_ShapForce_TrainSet.png'

def plot_cat_train_shap(path_train_summary, path_train_force):

    with col1:
        summary = st.image(path_train_summary)
    with col2:
        force = st.image(path_train_force)
    return summary, force

plot_cat_train_shap(path_train_summary, path_train_force)  

col1, col2 = st.columns(2)

# Test set
path_test_summary = path + '/catboost/results/Cat_ShapSummary_TestSet.png'

path_test_force = path + '/catboost/results/Cat_ShapForce_TestSet.png'

def plot_cat_test_shap(path_test_summary, path_test_force):

    with col1:
        summary = st.image(path_test_summary)
    with col2:
        force = st.image(path_test_force)
    return summary, force

plot_cat_test_shap(path_test_summary, path_test_force)  

###################################################################################################################
st.subheader('Models Metrics for Used Vehicle Price Prediction Using XGBoost', divider='blue')

y_train_pred = model_xgb.predict(dtrain)
y_test_pred = model_xgb.predict(dtest)

st.write('MAE train: %.3f, test: %.3f' % (
        mean_absolute_error(train_label, y_train_pred),
        mean_absolute_error(test_label, y_test_pred)))
st.write('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_label, y_train_pred),
        mean_squared_error(test_label, y_test_pred)))
st.write('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_label, y_train_pred, squared=False),
        mean_squared_error(test_label, y_test_pred, squared=False)))
st.write('R^2 train: %.3f, test: %.3f' % (
        r2_score(train_label, y_train_pred),
        r2_score(test_label, y_test_pred)))    

st.subheader('XGBoost: Feature Importance', divider='blue')

st.image(path + '/xgboost/results/XGB_FeatureImportance.png')

# SHAP
st.subheader('XGBoost: Model-based SHAP', divider='blue')

col1, col2 = st.columns(2)

# Train set
path_train_summary = path + '/xgboost/results/XGB_ShapSummary_TrainSet.png'

path_train_force = path + '/xgboost/results/XGB_ShapForce_TrainSet.png'

def plot_xgb_train_shap(path_train_summary, path_train_force):

    with col1:
        summary = st.image(path_train_summary)
    with col2:
        force = st.image(path_train_force)
    return summary, force

plot_xgb_train_shap(path_train_summary, path_train_force)  

col1, col2 = st.columns(2)

# Test set
path_test_summary = path + '/xgboost/results/XGB_ShapSummary_TestSet.png'

path_test_force = path + '/xgboost/results/XGB_ShapForce_TestSet.png'

def plot_xgb_test_shap(path_test_summary, path_test_force):

    with col1:
        summary = st.image(path_test_summary)
    with col2:
        force = st.image(path_test_force)
    return summary, force

plot_xgb_test_shap(path_test_summary, path_test_force)  

###################################################################################################################
link = 'Made by [Andrew Schultz](https://github.com/adataschultz/)'
st.markdown(link, unsafe_allow_html=True)