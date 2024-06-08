import os
import random
import numpy as np
import warnings
import pandas as pd
import mlflow
import lightgbm as lgb
from lightgbm import LGBMRegressor
import joblib
import dill as pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap 
warnings.filterwarnings('ignore')
my_dpi = 96

seed_value = 42
os.environ['usedCars_lgbmGPU'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def main():
    '''Main function of the script.'''

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.lightgbm.autolog()
    
    ###################
    #<prepare the data>
    ###################
    train_path = '../data/usedCars_trainSet.parquet.gzip'
    trainDF = pd.read_parquet(train_path)

    test_path = '../data/usedCars_testSet.parquet.gzip'
    testDF = pd.read_parquet(test_path)
        
    train_label = trainDF[['price']]
    train_features = trainDF.drop(columns=['price'])
    train_Features = pd.get_dummies(train_features, drop_first=True)

    test_label = testDF[['price']]
    test_features = testDF.drop(columns=['price'])
    test_Features = pd.get_dummies(test_features, drop_first=True)

    mlflow.log_metric('num_samples', train_Features.shape[0])
    mlflow.log_metric('num_features', train_Features.shape[1])

    print(f'Training with data of shape {train_Features.shape}')
    print(f'Testing with data of shape {test_Features.shape}')
    
    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    param = {
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device':'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbosity': -1,
        'n_estimators': 935,
        'learning_rate': 0.007999749230495415,
        'num_leaves': 352,
        'bagging_freq': 9,
        'subsample': 0.9499157527286387,
        'colsample_bytree': 0.776813542010267,
        'max_depth': 11,
        'lambda_l1': 0.026324983395628405,
        'lambda_l2': 0.001549095388756512,
        'min_child_samples': 541,
        'random_state': 42}
    
    mlflow.set_tag('usedcars_model', 'lightgbm')
    mlflow.set_tag('data scientist', 'ars')
    mlflow.log_params(param)

    usedcars_lgbm_model = LGBMRegressor(**param)

    usedcars_lgbm_model.fit(train_Features, train_label)

    print('\nModel Metrics for Used Cars LightGBM')
    y_train_pred = usedcars_lgbm_model.predict(train_Features)
    y_test_pred = usedcars_lgbm_model.predict(test_Features)

    train_mae = mean_absolute_error(train_label, y_train_pred)
    test_mae = mean_absolute_error(test_label, y_test_pred)
    train_mse = mean_squared_error(train_label, y_train_pred)
    test_mse = mean_squared_error(test_label, y_test_pred)
    train_rmse = mean_squared_error(train_label, y_train_pred, squared=False)
    test_rmse = mean_squared_error(test_label, y_test_pred, squared=False)
    train_r2 = r2_score(train_label, y_train_pred)
    test_r2 = r2_score(test_label, y_test_pred)

    mlflow.log_metric('train_mae', train_mae)
    mlflow.log_metric('train_mse', train_mse)
    mlflow.log_metric('train_rmse', train_rmse)
    mlflow.log_metric('train_r2', train_r2)
    mlflow.log_metric('test_mae', test_mae)
    mlflow.log_metric('test_mse', test_mse)
    mlflow.log_metric('test_rmse', test_rmse)
    mlflow.log_metric('test_r2', test_r2)

    print('MAE train: %.3f, test: %.3f' % (
            mean_absolute_error(train_label, y_train_pred),
            mean_absolute_error(test_label, y_test_pred)))
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(train_label, y_train_pred),
            mean_squared_error(test_label, y_test_pred)))
    print('RMSE train: %.3f, test: %.3f' % (
            mean_squared_error(train_label, y_train_pred, squared=False),
            mean_squared_error(test_label, y_test_pred, squared=False)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(train_label, y_train_pred),
            r2_score(test_label, y_test_pred)))
    
    ###################
    #</train the model>
    ###################

    ##########################
    #<save and register model>
    ##########################
     # Registering the model to the workspace
    print('Registering the model via MLFlow')
    mlflow.lightgbm.log_model(
        lgb_model=usedcars_lgbm_model,
        registered_model_name='usedcars_lgbm_model',
        artifact_path='usedcars_lgbm_model',
    )

    # Saving the model to a file
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open('./model/usedcars_lgbm_model.pkl', 'wb') as files: 
        pickle.dump(usedcars_lgbm_model, files)    

    ###########################
    #</save and register model>
    ###########################
    # Feature Importance
    lgb.plot_importance(usedcars_lgbm_model, max_num_features=15, figsize=(12,7))
    plt.tight_layout()
    plt.savefig('./results/LGBM_FeatureImportance.png', dpi=my_dpi*10, bbox_inches='tight');
    
    # SHAP   
    shap_explainer = shap.TreeExplainer(usedcars_lgbm_model)
    
    ex_filename = './model/LGBM_ShapExplainer.pkl'
    #shap_explainer = joblib.load(filename=ex_filename)
    joblib.dump(shap_explainer, filename=ex_filename, compress=('bz2', 9))

    shap_values_train = shap_explainer.shap_values(train_Features, test_label, check_additivity=False)

    shap_val_filename = './model/LGBM_shap_values_train.pkl'
    #shap_values_train = joblib.load(filename=shap_val_filename)
    joblib.dump(shap_values_train, filename=shap_val_filename, compress=('bz2', 9))

    plt.rcParams.update({'font.size': 1}) 
    fig = plt.figure(figsize=(1,1))
    shap.summary_plot(shap_values_train, train_Features, plot_size=[4,4], show=False)
    plt.title('Train Set: SHAP Summary Plot', y=1.3, fontsize=15)
    plt.tight_layout()
    plt.savefig('./results/LGBM_ShapSummary_TrainSet.png', dpi=my_dpi*10, bbox_inches='tight');

    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(10,10)) 
    shap.force_plot(shap_explainer.expected_value, shap_values_train[0,:], train_Features.iloc[0,:], 
                    matplotlib=True, show=False, figsize=(20,10))
    plt.title('Train Set: SHAP Force Plot', y=1.75, fontsize=35)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./results/LGBM_ShapForce_TrainSet.png', dpi=my_dpi*10, bbox_inches='tight');

    # Test set
    shap_values_test = shap_explainer.shap_values(test_Features, test_label, check_additivity=False)
    
    shap_val_filename = './model/LGBM_shap_values_test.pkl'
    joblib.dump(shap_values_test, filename=shap_val_filename, compress=('bz2', 9))
        
    #shap_val_filename = './model/LGBM_shap_values_test.pkl'
    #shap_values_test = joblib.load(filename=shap_val_filename)

    plt.rcParams.update({'font.size': 1})
    fig = plt.figure(figsize=(1,1))
    shap.summary_plot(shap_values_test, test_Features, plot_size=[4,4], show=False)
    plt.title('Test Set: SHAP Summary Plot', y=1.3, fontsize=15)
    plt.tight_layout()
    plt.savefig('./results/LGBM_ShapSummary_TestSet.png', dpi=my_dpi*10, bbox_inches='tight');
        
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(10,10)) 
    shap.force_plot(shap_explainer.expected_value, shap_values_test[0,:], test_Features.iloc[0,:], 
                    matplotlib=True, show=False, figsize=(20,10))
    plt.title('Test Set: SHAP Force Plot', y=1.75, fontsize=35)
    plt.tick_params(axis='x',labelsize=15)
    plt.tight_layout()
    plt.savefig('./results/LGBM_ShapForce_TestSet.png', dpi=my_dpi*10, bbox_inches='tight');
    
    # Stop Logging
    mlflow.end_run()

if __name__ == '__main__':
    main()