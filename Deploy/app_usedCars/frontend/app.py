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
from plotly.offline import iplot
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

seed_value = 42
os.environ['usedCars_GPU'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Path of the data
os.environ['TRAIN_DATA_DIR'] = '../data/usedCars_trainSet.csv'
os.environ['TEST_DATA_DIR'] = '../data/usedCars_testSet.csv'

# Load data
@st.cache_data
def load_data():
    try:
        train_data_path = os.environ['TRAIN_DATA_DIR']
        train_data = pd.read_csv(train_data_path)
        test_data_path = os.environ['TEST_DATA_DIR']
        test_data = pd.read_csv(test_data_path)
        return train_data, test_data
    except Exception as ex:
        raise(f'Error in loading file: {ex}', str(ex))
        
st.title('Pricing of Used Cars')

trainDF, testDF = load_data()

col1, col2 , col3 = st.columns(3)

with col2:    
# dict for the dataframes and their names
    dfs = {'Training Set' : trainDF, 'Test Set': testDF}

# plot the data
    fig = go.Figure()

    for i in dfs:
        fig = fig.add_trace(go.Box(x=dfs[i]['State'],
                                   y=dfs[i]['price'], 
                                   name=i))
        fig.update_layout(title='Train/Test Sets: Price of Vehicles in Different States')
    st.plotly_chart(fig)
    
col1, col2 , col3 = st.columns(3)

with col1:
    fig = px.pie(trainDF, values='year', names='State', title='Train Set: Vehicles Listed Per Year Per State')
    st.plotly_chart(fig)
    
with col2:
    fig = px.pie(testDF, values='year', names='State', title='Test Set: Vehicles Listed Per Year Per State')
    st.plotly_chart(fig)

col1, col2 , col3 = st.columns(3)

with col1:    
    fig = px.bar(trainDF, x='listed_date_yearMonth', y='price', color='State', labels={'y':'price'},
                 hover_data=['State'], title='Train Set: Total Price of Used Cars Per Month Per State')
    st.plotly_chart(fig)

with col2:    
    fig = px.bar(testDF, x='listed_date_yearMonth', y='price', color='State', labels={'y':'price'},
                 hover_data=['State'], title='Test Set: Total Price of Used Cars Per Month Per State')
    st.plotly_chart(fig)

col1, col2 , col3 = st.columns(3)

with col1:
    fig = px.box(trainDF, x='listing_color', y='price', points='all', title='Train Set: Price of Different Colored Vehicles')
    st.plotly_chart(fig)

with col2:
    fig = px.box(testDF, x='listing_color', y='price', points='all', title='Test Set: Price of Different Colored Vehicles')
    st.plotly_chart(fig)

col1, col2 , col3 = st.columns(3)

with col1:
    fig = px.bar(trainDF, x='State', y='price', color='listing_color', title='Train Set: Total Price of Different Colored Vehicles Per State')
    st.plotly_chart(fig)

with col2:
    fig = px.bar(testDF, x='State', y='price', color='listing_color', title='Test Set: Price of Different Colored Vehicles Per State')
    st.plotly_chart(fig)

col1, col2 , col3 = st.columns(3)

with col1:
    fig = px.bar(trainDF, x='State', y='daysonmarket', color='listing_color', title='Train Set: Number of Days on Market of Different Colored Vehicles Per State')
    st.plotly_chart(fig)

with col2:
    fig = px.bar(testDF, x='State', y='daysonmarket', color='listing_color', title='Test Set: Number of Days on Market of Different Colored Vehicles Per State')
    st.plotly_chart(fig)

###################################################################################################################
###################################################################################################################
st.subheader('Data Monitoring', divider='blue')

path_to_html = '../static/DataQualityPreset_report.html' 

with open(path_to_html,'r') as f: 
    data_quality = f.read()

###################################################################################################################
path_to_html = '../static/data_qualityTestPresets_report.html' 

with open(path_to_html,'r') as f: 
    data_quality_presets = f.read()
    
col1, col2 = st.columns(2)

with col1:
   st.subheader('Data Quality')
   st.components.v1.html(data_quality, scrolling=True, height=1000, width=1000)

with col2:
   st.components.v1.html(data_quality_presets, scrolling=True, height=1000, width=1000)
                         
###################################################################################################################
path_to_html = '../static/data_integrity_dataset_report_SummaryMissing.html' 

with open(path_to_html,'r') as f: 
    data_integrity = f.read()
                       
###################################################################################################################
path_to_html = '../static/data_stabilityTestPresets_report.html'

with open(path_to_html,'r') as f: 
    data_stability = f.read()

col1, col2 = st.columns(2)

with col1:
   st.subheader('Data Integrity')
   st.components.v1.html(data_integrity, scrolling=True, height=1000, width=1000)

with col2:
   st.subheader('Data Stability')
   st.components.v1.html(data_stability, scrolling=True, height=1000, width=1000)
                         
###################################################################################################################
path_to_html = '../static/data_drift_report.html'

with open(path_to_html,'r') as f: 
    data_drift = f.read()

col1, col2 , col3 = st.columns(3)

with col2:
    st.subheader('Data Drift')
    # Show in webpage
    st.components.v1.html(data_drift, scrolling=True, height=1000, width=1000)
                         
###################################################################################################################
###################################################################################################################
# Load models
os.environ['LGB_MODEL_DIR'] = '../lightgbm/model/usedcars_lgbm_model.pkl'
os.environ['CAT_MODEL_DIR'] = '../catboost/model/usedcars_cat_model'
os.environ['XGB_MODEL_DIR'] = '../xgboost/model/usedcars_xgb_model.bin'

def load_lgb_model():
    lgb_path = os.environ['LGB_MODEL_DIR']
    model = joblib.load(open(lgb_path,'rb'))
    return model

def load_cat_model():
    cat_path = os.environ['CAT_MODEL_DIR']
    model = CatBoostRegressor()
    model.load_model(cat_path)
    return model

def load_xgb_model():
    xgb_path = os.environ['XGB_MODEL_DIR']
    model = xgb.Booster()
    model.load_model(xgb_path)
    return model

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

st.image('../lightgbm/results/LGBM_FeatureImportance.png')

# SHAP
st.subheader('LightGBM: Model-based SHAP', divider='blue')

col1, col2 = st.columns(2)

with col1:
   st.image('../lightgbm/results/LGBM_ShapSummary_TrainSet.png')

with col2:
   st.image('../lightgbm/results/LGBM_ShapForce_TrainSet.png')

col1, col2 = st.columns(2)

with col1:
   st.image('../lightgbm/results/LGBM_ShapSummary_TestSet.png')

with col2:
   st.image('../lightgbm/results/LGBM_ShapForce_TestSet.png')

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

st.image('../catboost/results/Cat_FeatureImportance.png')

# SHAP
st.subheader('Catboost: Model-based SHAP', divider='blue')

col1, col2 = st.columns(2)

with col1:
   st.image('../catboost/results/Cat_ShapSummary_TrainSet.png')

with col2:
   st.image('../catboost/results/Cat_ShapForce_TrainSet.png')

col1, col2 = st.columns(2)

with col1:
   st.image('../catboost/results/Cat_ShapSummary_TestSet.png')

with col2:
   st.image('../catboost/results/Cat_ShapForce_TestSet.png')

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

st.image('../xgboost/results/XGB_FeatureImportance.png')

# SHAP
st.subheader('XGBoost: Model-based SHAP', divider='blue')

col1, col2 = st.columns(2)

with col1:
   st.image('../xgboost/results/XGB_ShapSummary_TrainSet.png')

with col2:
   st.image('../xgboost/results/XGB_ShapForce_TrainSet.png')

col1, col2 = st.columns(2)

with col1:
   st.image('../xgboost/results/XGB_ShapSummary_TestSet.png')

with col2:
   st.image('../xgboost/results/XGB_ShapForce_TestSet.png')

###################################################################################################################
st.subheader('Upload Data and Predict the Price Given the Features:', divider='blue')
# Set FastAPI endpoint
#endpoint = 'http://localhost:8000/predict'
#endpoint = 'http://172.17.0.1:8000/predict' # Specify this path for Dockerization
#endpoint = 'http://backend-api.default.svc.cluster.local:80/predict' # Specify this path for cloud
endpoint = 'https://app-usedcars-backend-6rtfkkrflq-uc.a.run.app/predict' # Specify this path for GCP cloud

test_csv = st.file_uploader('Choose a file', key=1)
if test_csv is not None:
    # To read file as bytes:
    bytes_data = test_csv.getvalue()

    # Can be used wherever a 'file-like' object is accepted:
    test_df = pd.read_csv(test_csv)

    st.subheader('Sample of Uploaded Dataset')
    st.write(test_df.head())

    # Convert dataframe to BytesIO object (for parsing as file into FastAPI later)
    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)  # write to BytesIO buffer
    test_bytes_obj.seek(0) # Reset pointer to avoid EmptyDataError

    files = {'file': ('test_dataset.csv', test_bytes_obj, 'multipart/form-data')}

    # Upon click of button
    if st.button('Start Prediction'):
        if len(test_df) == 0:
            st.write('Please upload a valid test dataset!')  # handle case with no data
        else:
            with st.spinner('Prediction in Progress. Please Wait...'):
                result = requests.post(endpoint, 
                                       files=files,
                                       timeout=8000)
            st.success('Success! 📥  Download button below to get prediction results')
            #print(result1)
            st.write(result)
            st.write(data=json.dumps(result.json()))
            st.download_button(
                label='Download',
                data=json.dumps(result.json()), # Download as JSON file object
                file_name='results_predictedPriceComparisons.json'
            )
            
###################################################################################################################
link = 'Made by [Andrew Schultz](https://github.com/adataschultz/)'
st.markdown(link, unsafe_allow_html=True)
