import streamlit as st
st.set_page_config(page_title='Predicting the Price of Used Vehicles',
                   layout='wide')
import os
import path
import sys
import gc
# import random
# import numpy as np
# import warnings
# import pandas as pd
import streamlit.components.v1 as components
#warnings.filterwarnings('ignore')

gc.collect()

# seed_value = 42
# os.environ['usedCars_GPU'] = str(seed_value)
# random.seed(seed_value)
# np.random.seed(seed_value)

# Set path
#path = '/mnt/UsedCars_Prices/Deploy/app_usedCars/frontend'
#path = '/mount/src/usedcars_prices/Deploy/app_usedCars/frontend'
#os.chdir(path)
dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)

st.markdown("<h1 style='text-align: center; color: black;'>Predicting the Price of Used Vehicles from CarGurus</h1>", unsafe_allow_html=True)

# Set number of columns
col1, col2 , col3 = st.columns(3)

# Price State: train/test
path_to_html = './static/traintest_priceState.html'

def plot_traintest_monthPriceState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        traintest_priceState = f.read()
    with col2:
        st.components.v1.html(traintest_priceState, height=500, width=700)
    return path_to_html

plot_traintest_monthPriceState(path_to_html)

del path_to_html, plot_traintest_monthPriceState

# Year State: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = './static/train_yearState.html'

def plot_train_yearState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_yearState = f.read()
    with col1:
        st.components.v1.html(train_yearState, height=500, width=700)
    return path_to_html

plot_train_yearState(path_to_html)

del path_to_html, plot_train_yearState

# Year State: Test
path_to_html = './static/test_yearState.html'

def plot_test_yearState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_yearState = f.read()
    with col2:
        st.components.v1.html(test_yearState, height=500, width=700)
    return path_to_html

plot_test_yearState(path_to_html)

del path_to_html, plot_test_yearState

gc.collect()

# Month Price State: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = './static/train_monthPriceState.html'

def plot_train_monthPriceState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_monthPriceState = f.read()
    with col1:
        st.components.v1.html(train_monthPriceState, height=500, width=700)
    return path_to_html

plot_train_monthPriceState(path_to_html)

del path_to_html, plot_train_monthPriceState

# Month Price State: Test
path_to_html = './static/test_monthPriceState.html'

def plot_test_monthPriceState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_monthPriceState = f.read()
    with col2:
        st.components.v1.html(test_monthPriceState, height=500, width=700)
    return path_to_html

plot_test_monthPriceState(path_to_html)

del path_to_html, plot_test_monthPriceState

gc.collect()

# Price Color: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = './static/train_priceColor.html'

def plot_train_priceColor(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_priceColor = f.read()
    with col1:
        st.components.v1.html(train_priceColor, height=500, width=700)
    return path_to_html

plot_train_priceColor(path_to_html)

del path_to_html, plot_train_priceColor

# Price Color: Test
path_to_html = './static/test_priceColor.html'

def plot_test_priceColor(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_priceColor = f.read()
    with col2:
        st.components.v1.html(test_priceColor, height=500, width=700)
    return path_to_html

plot_test_priceColor(path_to_html)

del path_to_html, plot_test_priceColor

gc.collect()

# Price Color State: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = './static/train_priceColorState.html'

def plot_train_PriceColorState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_PriceColorState = f.read()
    with col1:
        st.components.v1.html(train_PriceColorState, height=500, width=700)
    return path_to_html

plot_train_PriceColorState(path_to_html)

del path_to_html, plot_train_PriceColorState

# Price Color State: Test
path_to_html = './static/test_priceColorState.html'

def plot_test_PriceColorState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_PriceColorState = f.read()
    with col2:
        st.components.v1.html(test_PriceColorState, height=500, width=700)
    return path_to_html

plot_test_PriceColorState(path_to_html)

del path_to_html, plot_test_PriceColorState

gc.collect()

# Days on Market Color State: Train
col1, col2  = st.columns(2, gap='small')

path_to_html = './static/train_domColorState.html'

def plot_train_domColorState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        train_domColorState = f.read()
    with col1:
        st.components.v1.html(train_domColorState, height=500, width=700)
    return path_to_html

plot_train_domColorState(path_to_html)

del path_to_html, plot_train_domColorState

# Days on Market Color State: Test
path_to_html = './static/test_domColorState.html'

def plot_test_domColorState(path_to_html):
    
    with open(path_to_html,'r') as f: 
        test_domColorState = f.read()
    with col2:
        st.components.v1.html(test_domColorState, height=500, width=700)
    return path_to_html

plot_test_domColorState(path_to_html)

del path_to_html, plot_test_domColorState

gc.collect()

###################################################################################################################
###################################################################################################################
st.subheader('Data Monitoring', divider='blue')

# Data Quality
path_to_html_quality = './static/DataQualityPreset_report.html' 

path_to_html_presets = './static/data_qualityTestPresets_report.html' 

col1, col2 = st.columns(2)

def plot_dataQuality(path_to_html_quality): 
    with open(path_to_html_quality,'r') as f: 
        data_quality = f.read()

    with col1:
        st.subheader('Data Quality')
        st.components.v1.html(data_quality, scrolling=True, height=900, width=900)
    return data_quality

plot_dataQuality(path_to_html_quality)

del path_to_html_quality, plot_dataQuality

gc.collect()

def plot_dataQualityPresets(path_to_html_presets):
    
    with open(path_to_html_presets,'r') as f: 
        data_quality_presets = f.read()

    with col2:
        st.components.v1.html(data_quality_presets, scrolling=True, height=900, width=900)
    return data_quality_presets

plot_dataQualityPresets(path_to_html_presets)

del path_to_html_presets, plot_dataQualityPresets

gc.collect()

###################################################################################################################
# Data Integrity and Stability
path_to_html_integrity = './static/data_integrity_dataset_report_SummaryMissing.html' 

path_to_html_stability = './static/data_stabilityTestPresets_report.html'

col1, col2 = st.columns(2)

def plot_dataIntegrity(path_to_html_integrity):
    
    with open(path_to_html_integrity,'r') as f: 
        data_integrity = f.read()

    with col1:
        st.subheader('Data Integrity')
        st.components.v1.html(data_integrity, scrolling=True, height=900, width=900)
    
    return data_integrity
    
plot_dataIntegrity(path_to_html_integrity)

del path_to_html_integrity, plot_dataIntegrity

gc.collect()

def plot_dataStabilty(path_to_html_stability):
    
    with open(path_to_html_stability,'r') as f: 
        data_stability = f.read()
    with col2:
        st.subheader('Data Stability')
        st.components.v1.html(data_stability, scrolling=True, height=900, width=900)
    
    return data_stability

plot_dataStabilty(path_to_html_stability)

del path_to_html_stability, plot_dataStabilty

gc.collect()

###################################################################################################################
# Data Drift
path_to_html_drift = './static/data_drift_report.html'

col1, col2 , col3 = st.columns(3)

def plot_dataDrift(path_to_html_drift):

    with open(path_to_html_drift,'r') as f: 
        data_drift = f.read()
    with col2:
        st.subheader('Data Drift')
        st.components.v1.html(data_drift, scrolling=True, height=900, width=900) 
    return data_drift

data_drift = plot_dataDrift(path_to_html_drift)  

del path_to_html_drift, plot_dataDrift

gc.collect()

###################################################################################################################
###################################################################################################################
st.subheader('LightGBM: Feature Importance', divider='blue')

path_lgbm_importance = './lightgbm/results/LGBM_FeatureImportance.png'

def plot_lgb_importance(path_lgbm_importance):

    feat_importance = st.image(path_lgbm_importance)

    return feat_importance

plot_lgb_importance(path_lgbm_importance)

del path_lgbm_importance, plot_lgb_importance

# SHAP
st.subheader('LightGBM: Model-based SHAP', divider='blue')

col1, col2 = st.columns(2)

# Train set
path_train_summary = './lightgbm/results/LGBM_ShapSummary_TrainSet.png'

path_train_force = './lightgbm/results/LGBM_ShapForce_TrainSet.png'

def plot_lgb_train_shap(path_train_summary, path_train_force):

    with col1:
        summary = st.image(path_train_summary)
    with col2:
        force = st.image(path_train_force)
    return summary, force

plot_lgb_train_shap(path_train_summary, path_train_force)  

del path_train_summary, path_train_force, plot_lgb_train_shap

# Test set
col1, col2 = st.columns(2)

path_test_summary = './lightgbm/results/LGBM_ShapSummary_TestSet.png'

path_test_force = './lightgbm/results/LGBM_ShapForce_TestSet.png'

def plot_lgb_test_shap(path_test_summary, path_test_force):

    with col1:
        summary = st.image(path_test_summary)
    with col2:
        force = st.image(path_test_force)
    return summary, force

plot_lgb_test_shap(path_test_summary, path_test_force)

del path_test_summary, path_test_force, plot_lgb_test_shap

gc.collect()

###################################################################################################################
st.subheader('Catboost: Feature Importance', divider='blue')

path_cat_importance = './catboost/results/Cat_FeatureImportance.png'

def plot_cat_importance(path_cat_importance):

    feat_importance = st.image(path_cat_importance)

    return feat_importance

plot_cat_importance(path_cat_importance)

del path_cat_importance, plot_cat_importance

# SHAP
st.subheader('Catboost: Model-based SHAP', divider='blue')

col1, col2 = st.columns(2)

# Train set
path_train_summary = './catboost/results/Cat_ShapSummary_TrainSet.png'

path_train_force = './catboost/results/Cat_ShapForce_TrainSet.png'

def plot_cat_train_shap(path_train_summary, path_train_force):

    with col1:
        summary = st.image(path_train_summary)
    with col2:
        force = st.image(path_train_force)
    return summary, force

plot_cat_train_shap(path_train_summary, path_train_force)  

del path_train_summary, path_train_force, plot_cat_train_shap

col1, col2 = st.columns(2)

# Test set
path_test_summary = './catboost/results/Cat_ShapSummary_TestSet.png'

path_test_force = './catboost/results/Cat_ShapForce_TestSet.png'

def plot_cat_test_shap(path_test_summary, path_test_force):

    with col1:
        summary = st.image(path_test_summary)
    with col2:
        force = st.image(path_test_force)
    return summary, force

plot_cat_test_shap(path_test_summary, path_test_force)  

del path_test_summary, path_test_force, plot_cat_test_shap

gc.collect()

###################################################################################################################
st.subheader('XGBoost: Feature Importance', divider='blue')

path_xgb_importance = './xgboost/results/XGB_FeatureImportance.png'

def plot_xgb_importance(path_xgb_importance):

    feat_importance = st.image(path_xgb_importance)

    return feat_importance

plot_xgb_importance(path_xgb_importance)

del path_xgb_importance, plot_xgb_importance

# SHAP
st.subheader('XGBoost: Model-based SHAP', divider='blue')

col1, col2 = st.columns(2)

# Train set
path_train_summary = './xgboost/results/XGB_ShapSummary_TrainSet.png'

path_train_force = './xgboost/results/XGB_ShapForce_TrainSet.png'

def plot_xgb_train_shap(path_train_summary, path_train_force):

    with col1:
        summary = st.image(path_train_summary)
    with col2:
        force = st.image(path_train_force)
    return summary, force

plot_xgb_train_shap(path_train_summary, path_train_force)  

del path_train_summary, path_train_force, plot_xgb_train_shap

col1, col2 = st.columns(2)

# Test set
path_test_summary = './xgboost/results/XGB_ShapSummary_TestSet.png'

path_test_force = './xgboost/results/XGB_ShapForce_TestSet.png'

def plot_xgb_test_shap(path_test_summary, path_test_force):

    with col1:
        summary = st.image(path_test_summary)
    with col2:
        force = st.image(path_test_force)
    return summary, force

plot_xgb_test_shap(path_test_summary, path_test_force)  

del path_test_summary, path_test_force, plot_xgb_test_shap

gc.collect()

###################################################################################################################
link = 'Made by [Andrew Schultz](https://github.com/adataschultz/)'
st.markdown(link, unsafe_allow_html=True)