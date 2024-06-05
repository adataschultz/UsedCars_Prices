import streamlit as st
st.set_page_config(page_title='Predicting the Price of Used Vehicles',
                   layout='wide')
import os
import random
import numpy as np
import warnings
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.io as pio
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
import requests
from requests import ConnectionError
import json
import io
from io import StringIO, BytesIO
warnings.filterwarnings('ignore')
pio.templates.default = 'plotly'

seed_value = 42
os.environ['usedCars_GPU'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Set path
#path = '/mnt/UsedCars_Prices/Deploy/app_usedCars/frontend'
path = '/mount/src/usedcars_prices/Deploy/app_usedCars/frontend'
os.chdir(path)
        
# Load data
@st.cache_data
def load_data():
    try:
        train_data = pd.read_parquet(path + '/data/usedCars_trainSet.parquet.gzip')
        test_data = pd.read_parquet(path + '/data/usedCars_testSet.parquet.gzip')
        return train_data, test_data
    except Exception as ex:
        raise(f'Error in loading file: {ex}', str(ex))
        
st.markdown("<h1 style='text-align: center; color: black;'>Predicting the Price of Used Vehicles from CarGurus</h1>", unsafe_allow_html=True)

# Load data
trainDF, testDF = load_data()

# Sort by state
cat = ['CA','NY','FL','TX','PA','OH','IL']

trainDF['State'] = pd.Categorical(trainDF['State'], categories=cat)
trainDF.sort_values(['State'], inplace=True)
trainDF.reset_index(drop=True, inplace=True)

testDF['State'] = pd.Categorical(testDF['State'], categories=cat)
testDF.sort_values(['State'], inplace=True)
testDF.reset_index(drop=True, inplace=True)

col1, col2 , col3 = st.columns(3)

# Price State: train/test
@st.cache_resource(show_spinner=False)
def plot_traintest_priceState(train, test):
    dfs = {'Training Set': trainDF, 'Test Set': testDF}

    fig = go.Figure()

    for i in dfs:
        fig = fig.add_trace(go.Box(x=dfs[i]['State'],
                                   y=dfs[i]['price'], 
                                   name=i))
    fig.update_layout(title={'text': 'Train/Test Sets: Price of Vehicles in Different States','y': 0.9,'x': 0.5, 'xanchor': 'center','yanchor': 'top'}, titlefont=dict(size=24))
    fig.update_layout(legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title='Location (State)', title_font_color='black', titlefont=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title='Price (USD)', title_font_color='black', titlefont=dict(size=18), tickfont=dict(size=18, color='black'))
    f = open(path + '/static/traintest_priceState.html', 'w')
    f.close()
    with open(path + '/static/traintest_priceState.html', 'a') as f:
        f.write(fig.to_html(full_html=True))
    f.close()
    return st.plotly_chart(fig)

with col2:
    plot_traintest_priceState(trainDF, testDF)
    
col1, col2  = st.columns(2, gap='small')

# Year State: Train/Test
@st.cache_resource(show_spinner=False)
def plot_train_yearState(train):
    fig = px.pie(train, values='year', names='State', color='State', color_discrete_map={'CA': '#EB663B',
                                                                                         'NY': '#1CA71C', 
                                                                                         'FL': '#2E91E5',
                                                                                         'TX': '#AF0038',
                                                                                         'PA': '#511CFB',
                                                                                         'OH': '#00A08B', 
                                                                                         'IL': '#EBE426'},
                 title='Train Set: Vehicles Listed Per Year Per State')
    fig.update_layout(title={'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, titlefont=dict(size=24))
    fig.update_layout(legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    f = open(path + '/static/train_yearState.html', 'w')
    f.close()
    with open(path + '/static/train_yearState.html', 'a') as f:
            f.write(fig.to_html(full_html=True))
    f.close()    
    return st.plotly_chart(fig)

with col1:
    plot_train_yearState(trainDF)
    
@st.cache_resource(show_spinner=False)
def plot_test_yearState(test):
    fig = px.pie(test, values='year', names='State', color='State', color_discrete_map={'CA': '#EB663B',
                                                                                        'NY': '#1CA71C', 
                                                                                        'FL': '#2E91E5',
                                                                                        'TX': '#AF0038',
                                                                                        'PA': '#511CFB',
                                                                                        'OH': '#00A08B', 
                                                                                        'IL': '#EBE426'}, 
                 title='Test Set: Vehicles Listed Per Year Per State')
    fig.update_layout(title={'y': 0.9,'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, titlefont=dict(size=24))
    fig.update_layout(legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    f = open(path + '/static/test_yearState.html', 'w')
    f.close()
    with open(path + '/static/test_yearState.html', 'a') as f:
            f.write(fig.to_html(full_html=True))
    f.close()
    return st.plotly_chart(fig)
    
with col2:
    plot_test_yearState(testDF)

col1, col2  = st.columns(2, gap='small')

# Month Price State: Train/Test
@st.cache_resource(show_spinner=False)
def plot_train_monthPriceState(train):
    fig = px.bar(trainDF, x='listed_date_yearMonth', y='price', color='State', labels={'y':'price'},
                 hover_data=['State'],
                 color_discrete_sequence=['#EB663B','#1CA71C', '#2E91E5','#AF0038', '#511CFB','#00A08B', '#EBE426'],
                 title='Train Set: Total Price of Used Vehicles Per Month Per State')
    fig.update_layout(title={'y': 0.9,'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, titlefont=dict(size=24))
    fig.update_layout(legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title='Month of the Year Listed', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title='Price (USD)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    fig.update_traces(dict(marker_line_width=0))
    f = open(path + '/static/train_monthPriceState.html', 'w')
    f.close()
    with open(path + '/static/train_monthPriceState.html', 'a') as f:
            f.write(fig.to_html())
    f.close()              
    return st.plotly_chart(fig)

with col1:    
    plot_train_monthPriceState(trainDF)

@st.cache_resource(show_spinner=False)
def plot_test_monthPriceState(test):
    fig = px.bar(testDF, x='listed_date_yearMonth', y='price', color='State', labels={'y':'price'},
                 hover_data=['State'],
                 color_discrete_sequence=['#EB663B','#1CA71C', '#2E91E5','#AF0038', '#511CFB','#00A08B', '#EBE426'],
                 title='Test Set: Total Price of Used Vehicles Per Month Per State')
    fig.update_layout(title={'y': 0.9,'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, titlefont=dict(size=24))
    fig.update_layout(legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title='Month of the Year Listed', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title='Price (USD)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')    
    fig.update_traces(dict(marker_line_width=0))
    f = open(path + '/static/test_monthPriceState.html', 'w')
    f.close()
    with open(path + '/static/test_monthPriceState.html', 'a') as f:
            f.write(fig.to_html())
    f.close()               
    return st.plotly_chart(fig)

with col2:
    plot_test_monthPriceState(testDF)

# Sort listing_color with State
cat = ['BLACK','BLUE','GRAY','RED','SILVER','WHITE'] 

trainDF['listing_color'] = pd.Categorical(trainDF['listing_color'], categories=cat)
trainDF.sort_values(['listing_color', 'State'], inplace=True)
trainDF.reset_index(drop=True, inplace=True)

testDF['listing_color'] = pd.Categorical(testDF['listing_color'], categories=cat)
testDF.sort_values(['listing_color', 'State'], inplace=True)
testDF.reset_index(drop=True, inplace=True)

col1, col2  = st.columns(2, gap='small')

# Price Color: Train/Test
@st.cache_resource(show_spinner=False)
def plot_train_priceColor(train):
    fig = px.box(trainDF, y='price', x='listing_color', color='listing_color', 
                 color_discrete_sequence=['#222A2A', '#2E91E5','#565656','#AF0038', '#778AAE','#E2E2E2'],
                 title='Train Set: Price of Different Colored Vehicles')
    fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, titlefont=dict(size=24))
    fig.update_layout(legend_title='Color', legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18) 
    fig.update_xaxes(title='Vehicle Color', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title='Price (USD)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    fig.update_traces(dict(marker_line_width=0))
    f = open(path + '/static/train_priceColor.html', 'w')
    f.close()
    with open(path + '/static/train_priceColor.html', 'a') as f:
            f.write(fig.to_html(full_html=True))
    f.close()                
    return st.plotly_chart(fig)

with col1:
    plot_train_priceColor(trainDF)

@st.cache_resource(show_spinner=False)
def plot_test_priceColor(test):
    fig = px.box(testDF, y='price', x='listing_color', color='listing_color', 
                 color_discrete_sequence=['#222A2A', '#2E91E5','#565656','#AF0038', '#778AAE','#E2E2E2'],
                 title='Test Set: Price of Different Colored Vehicles')
    fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, titlefont=dict(size=24))
    fig.update_layout(legend_title='Color', legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title='Vehicle Color', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title='Price (USD)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide') 
    fig.update_traces(dict(marker_line_width=0))
    f = open(path + '/static/test_priceColor.html', 'w')
    f.close()
    with open(path + '/static/test_priceColor.html', 'a') as f:
            f.write(fig.to_html(full_html=True))
    f.close()              
    return st.plotly_chart(fig)

with col2:
    plot_test_priceColor(testDF)

col1, col2  = st.columns(2, gap='small')

# Price Color State: Train/Test
@st.cache_resource(show_spinner=False)
def plot_train_priceColorState(train):
    fig = px.bar(trainDF, y='price', x='State', color='listing_color', 
                 color_discrete_sequence=['#222A2A', '#2E91E5','#565656','#AF0038', '#778AAE','#E2E2E2'],
                 title='Train Set: Total Price of Different Colored Vehicles Per State')
    fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, titlefont=dict(size=24))
    fig.update_layout(legend_title='Color', legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title='Location (State)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title='Price (USD)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    fig.update_traces(dict(marker_line_width=0))
    f = open(path + '/static/train_priceColorState.html', 'w')
    f.close()
    with open(path + '/static/train_priceColorState.html', 'a') as f:
            f.write(fig.to_html())
    f.close()              
    return st.plotly_chart(fig)

with col1:
    plot_train_priceColorState(trainDF)

@st.cache_resource(show_spinner=False)
def plot_test_priceColorState(test):
    fig = px.bar(testDF, y='price', x='State', color='listing_color', 
                 color_discrete_sequence=['#222A2A', '#2E91E5','#565656','#AF0038', '#778AAE','#E2E2E2'],
                 title='Test Set: Total Price of Different Colored Vehicles Per State')
    fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, titlefont=dict(size=24))
    fig.update_layout(legend_title='Color', legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title='Location (State)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title='Price (USD)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    fig.update_traces(dict(marker_line_width=0))
    f = open(path + '/static/test_priceColorState.html', 'w')
    f.close()
    with open(path + '/static/test_priceColorState.html', 'a') as f:
            f.write(fig.to_html())
    f.close()                  
    return st.plotly_chart(fig)

with col2:
    plot_test_priceColorState(testDF)

col1, col2  = st.columns(2, gap='small')

# Days on Market Color State: Train/Test
@st.cache_resource(show_spinner=False)
def plot_train_domColorState(train):
    fig = px.bar(trainDF, y='daysonmarket', x='State', color='listing_color', 
                 color_discrete_sequence=['#222A2A', '#2E91E5','#565656','#AF0038', '#778AAE','#E2E2E2'],                
                 title='Train Set: Number of Days on Market of Different Colored Vehicles Per State')
    fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, titlefont=dict(size=20))
    fig.update_layout(legend_title='Color', legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title='Location (State)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title='Duration of Time on Market (Days)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    fig.update_traces(dict(marker_line_width=0))
    f = open(path + '/static/train_domColorState.html', 'w')
    f.close()
    with open(path + '/static/train_domColorState.html', 'a') as f:
            f.write(fig.to_html())
    f.close()                
    return st.plotly_chart(fig)

with col1:
    plot_train_domColorState(trainDF)

@st.cache_resource(show_spinner=False)
def plot_test_domColorState(test):
    fig = px.bar(testDF, y='daysonmarket', x='State', color='listing_color',
                 color_discrete_sequence=['#222A2A', '#2E91E5','#565656','#AF0038', '#778AAE','#E2E2E2'],
                 title='Test Set: Number of Days on Market of Different Colored Vehicles Per State')
    fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, titlefont=dict(size=20))
    fig.update_layout(legend_title='Color', legend_title_font_color='black', legend_title_font_size=18, legend_font_size=18)
    fig.update_xaxes(title='Location (State)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_yaxes(title='Duration of Time on Market (Days)', title_font_color='black', title_font=dict(size=18), tickfont=dict(size=16, color='black'))
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    fig.update_traces(dict(marker_line_width=0))
    f = open(path + '/static/test_domColorState.html', 'w')
    f.close()
    with open(path + '/static/test_domColorState.html', 'a') as f:
            f.write(fig.to_html())
    f.close()                    
    return st.plotly_chart(fig)

with col2:
    plot_test_domColorState(testDF)

###################################################################################################################
###################################################################################################################
st.subheader('Data Monitoring', divider='blue')

# Data Quality
path_to_html_quality = path + '/static/DataQualityPreset_report.html' 

path_to_html_presets = path + '/static/data_qualityTestPresets_report.html' 

col1, col2 = st.columns(2)

@st.cache_resource(show_spinner=False)
def plot_dataQuality(path_to_html_quality): 
    with open(path_to_html_quality,'r') as f: 
        data_quality = f.read()

    with col1:
        st.subheader('Data Quality')
        st.components.v1.html(data_quality, scrolling=True, height=900, width=900)
    return data_quality

plot_dataQuality(path_to_html_quality)

@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
def plot_dataIntegrity(path_to_html_integrity):
    
    with open(path_to_html_integrity,'r') as f: 
        data_integrity = f.read()

    with col1:
        st.subheader('Data Integrity')
        st.components.v1.html(data_integrity, scrolling=True, height=900, width=900)
    
    return data_integrity
    
plot_dataIntegrity(path_to_html_integrity)

@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
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
# Load models
os.environ['LGB_MODEL_DIR'] = path + '/lightgbm/model/usedcars_lgbm_model.pkl'
os.environ['CAT_MODEL_DIR'] = path + '/catboost/model/usedcars_cat_model'
os.environ['XGB_MODEL_DIR'] = path + '/xgboost/model/usedcars_xgb_model.bin'

@st.cache_resource(show_spinner=False)
def load_lgb_model():
    lgb_path = os.environ['LGB_MODEL_DIR']
    model = joblib.load(open(lgb_path,'rb'))
    return model

@st.cache_resource(show_spinner=False)
def load_cat_model():
    cat_path = os.environ['CAT_MODEL_DIR']
    model = CatBoostRegressor()
    model.load_model(cat_path)
    return model

@st.cache_resource(show_spinner=False)
def load_xgb_model():
    xgb_path = os.environ['XGB_MODEL_DIR']
    model = xgb.Booster()
    model.load_model(xgb_path)
    return model

# Prepare data
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

# Train set
path_train_summary = path + '/lightgbm/results/LGBM_ShapSummary_TrainSet.png'

path_train_force = path + '/lightgbm/results/LGBM_ShapForce_TrainSet.png'

@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
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