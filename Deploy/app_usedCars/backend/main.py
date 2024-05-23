# Import Needed Libraries
import os
import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import joblib
import io
from io import StringIO
from io import BytesIO

#uvicorn  main:app --port 8050

# Create the FastAPI application
app = FastAPI(title='Used Car Price Prediction API', version='1.0',
              description='Regression modeling techniques are used for prediction')

categorical_features_indices = ['body_type', 'fuel_type', 'listing_color','transmission', 'wheel_system_display', 'State','listed_date_yearMonth', 'is_new']

LGB_MODEL_DIR = '../lightgbm/model/usedcars_lgbm_model.pkl'
CAT_MODEL_DIR = '../catboost/model/usedcars_cat_model'
XGB_MODEL_DIR = '../xgboost/model/usedcars_xgb_model.bin'

# Path to the model
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

# Load models
model_lgb = load_lgb_model()
model_cat = load_cat_model()
model_xgb = load_xgb_model()

@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to the End to End Used Cars Project</h2>
    <p> The model and FastAPI instances have been set up successfully </p>
    <p> You can view the FastAPI UI by heading to localhost:8000 </p>
    <p> Proceed to initialize the Streamlit UI (frontend/app.py) to submit prediction requests </p>
    </body>
    """
    return HTMLResponse(content=content)

# ML API endpoint for making prediction aganist the request received from client
# Create POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: bytes = File(...)):
    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)

    price = test_df[['price']]
    test_df = test_df.drop(['price'], axis=1)
    
    testDF = pd.get_dummies(test_df, drop_first=True)
    
    # Get the model's features in the correct order
    boost = model_lgb.booster_
    cols = boost.feature_name()

    # Use col to reindex the prediction DataFrame
    test = testDF.reindex(columns=cols) # -> df now has the same col ordering as the model
    
    categorical_features_indices = ['body_type', 'fuel_type', 'listing_color','transmission', 'wheel_system_display', 'State','listed_date_yearMonth', 'is_new']

    dtest = xgb.DMatrix(testDF, label=price)

    # Generate predictions with best model (output is H2O frame)
    preds_lgb = model_lgb.predict(test)
    preds_lgb = pd.DataFrame(preds_lgb.T, columns=['predicted_price'])
    df_lgb = pd.concat([price, preds_lgb], axis=1)
    df_lgb['predicted_difference'] = df_lgb['price'] - df_lgb['predicted_price']
    df_lgb['predicted_percentageDiff'] = (df_lgb['predicted_difference'] / df_lgb['price']) * 100
    df_lgb['algorithm'] = 'LightGBM'

    preds_cat = model_cat.predict(test_df)
    preds_cat = pd.DataFrame(preds_cat.T, columns=['predicted_price'])
    df_cat = pd.concat([price, preds_cat], axis=1)
    df_cat['predicted_difference'] = df_cat['price'] - df_cat['predicted_price']
    df_cat['predicted_percentageDiff'] = (df_cat['predicted_difference'] / df_cat['price']) * 100
    df_cat['algorithm'] = 'Catboost'

    preds_xgb = model_xgb.predict(dtest)
    preds_xgb = pd.DataFrame(preds_xgb.T, columns=['predicted_price'])
    df_xgb = pd.concat([price, preds_xgb], axis=1)
    df_xgb['predicted_difference'] = df_xgb['price'] - df_xgb['predicted_price']
    df_xgb['predicted_percentageDiff'] = (df_xgb['predicted_difference'] / df_xgb['price']) * 100
    df_xgb['algorithm'] = 'XGBoost'
    
    df = np.concatenate([df_lgb, df_cat, df_xgb])
    print(df)
    
    preds_final = pd.DataFrame(df, columns=['price', 'predicted_price', 'predicted_difference', 'predicted_percentageDiff', 'algorithm'])
    preds_final = preds_final.to_dict()
    print(preds_final)
    
    # Convert predictions into JSON format
    json_compatible_item_data = jsonable_encoder(preds_final)
    return JSONResponse(content=json_compatible_item_data)