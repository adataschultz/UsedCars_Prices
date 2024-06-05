import os
import random
import numpy as np
import warnings
import pandas as pd
import mlflow
import mlflow.xgboost
import joblib
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
warnings.filterwarnings('ignore')
my_dpi = 96

seed_value = 42
os.environ['usedCars_xgbGPU'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def main():
    '''Main function of the script.'''

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.xgboost.autolog()

    ###################
    #<prepare the data>
    ###################
    train_path = '../data/usedCars_trainSet.parquet.gzip'
    trainDF = pd.read_parquet(train_path) 

    test_path = '../data/usedCars_testSet.parquet.gzip'
    testDF = pd.read_parquet(test_path)

    train_label = trainDF[['price']]
    test_label = testDF[['price']]

    train_features = trainDF.drop(['price'], axis=1)
    test_features = testDF.drop(['price'], axis=1)

    train_Features = pd.get_dummies(train_features, drop_first=True)
    test_Features = pd.get_dummies(test_features, drop_first=True)

    features_train = train_Features.columns.tolist()
    features_test = test_Features.columns.tolist()
    
    dtrain = xgb.DMatrix(train_Features, label=train_label)
    dtest = xgb.DMatrix(test_Features, label=test_label)

    mlflow.log_metric('num_samples', train_Features.shape[0])
    mlflow.log_metric('num_features', train_Features.shape[1])

    print(f'Training with data of shape {train_Features.shape}')

    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    num_round = 500
    param = {'objective': 'reg:squarederror',
             'metric': 'rmse',
             'booster': 'gbtree',
             'tree_method': 'hist',
             'device': 'cuda',
             'scale_pos_weight': 1,
             'use_label_encoder': False,
             'random_state': 42,
             'verbosity': 0,
             'n_estimators': 549,
             'max_depth': 14,
             'subsample': 0.7997070496461064,
             'gamma': 2.953865805049196e-05,
             'learning_rate': 0.04001808814037916,
             'reg_alpha': 0.018852758055925938,
             'reg_lambda': 1.8216639376033342e-06,
             'colsample_bytree': 0.56819205236003,
             'colsample_bylevel': 0.5683397007952175,
             'min_child_weight': 7}

    usedcars_xgb_model = xgb.train(param, dtrain, num_round)

    print('\nModel Metrics for Used Cars XGBoost')
    y_train_pred = usedcars_xgb_model.predict(dtrain)
    y_test_pred = usedcars_xgb_model.predict(dtest)

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
    mlflow.xgboost.log_model(
        xgb_model=usedcars_xgb_model,
        registered_model_name='usedcars_xgb_model',
        artifact_path='usedcars_xgb_model',
    )

    # Saving the model to a file
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    usedcars_xgb_model.save_model( './model/usedcars_xgb_model.bin')

    ###########################
    #</save and register model>
    ###########################
    usedcars_xgb_model.feature_names = features_train
    
    # Feature importance
    fig = plt.figure(figsize=(4,6))
    plot_importance(usedcars_xgb_model, max_num_features=15);
    plt.savefig('./results/XGB_FeatureImportance.png',
                dpi=my_dpi*10, bbox_inches='tight');

    # SHAP   
    shap_explainer = shap.TreeExplainer(usedcars_xgb_model)
    
    ex_filename = './model/XGB_ShapExplainer.pkl'
    joblib.dump(shap_explainer, filename=ex_filename, compress=('bz2', 9))

    #ex_filename = './model/XGB_ShapExplainer.pkl'
    #shap_explainer = joblib.load(filename=ex_filename)

    shap_values_train = shap_explainer.shap_values(train_Features, train_label.values, check_additivity=False)

    shap_val_filename = './model/XGB_shap_values_train.pkl'
    joblib.dump(shap_values_train, filename=shap_val_filename, compress=('bz2', 9))

    #shap_val_filename = './model/XGB_shap_values_train.pkl'
    #shap_values_train = joblib.load(filename=shap_val_filename)

    plt.rcParams.update({'font.size': 1})
    fig = plt.figure(figsize=(1,1))
    shap.summary_plot(shap_values_train, train_Features, plot_size=[4,4], show=False)
    plt.title('Train Set: SHAP Summary Plot', y=1.3, fontsize=15)
    plt.tight_layout()
    plt.savefig('./results/XGB_ShapSummary_TrainSet.png', dpi=my_dpi*10, bbox_inches='tight');

    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(10,10))
    shap.force_plot(shap_explainer.expected_value, shap_values_train[0,:], train_Features.iloc[0,:], 
                    matplotlib=True, show=False, figsize=(20,10))
    plt.title('Train Set: SHAP Force Plot', y=1.75, fontsize=35)
    plt.tick_params(axis='x',labelsize=15)
    plt.tight_layout()
    plt.savefig('./results/XGB_ShapForce_TrainSet.png', dpi=my_dpi*10, bbox_inches='tight');

    # Test set
    shap_values_test = shap_explainer.shap_values(test_Features, test_label, check_additivity=False)
    
    shap_val_filename = './model/XGB_shap_values_test.pkl'
    joblib.dump(shap_values_test, filename=shap_val_filename, compress=('bz2', 9))
        
    #shap_val_filename = './model/XGB_shap_values_test.pkl'
    #shap_values_test = joblib.load(filename=shap_val_filename)

    plt.rcParams.update({'font.size': 1})
    fig = plt.figure(figsize=(1,1))
    shap.summary_plot(shap_values_test, test_Features, plot_size=[4,4], show=False)
    plt.title('Test Set: SHAP Summary Plot', y=1.3, fontsize=15)
    plt.tight_layout()
    plt.savefig('./results/XGB_ShapSummary_TestSet.png', dpi=my_dpi*10, bbox_inches='tight');

    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(10,10))
    shap.force_plot(shap_explainer.expected_value, shap_values_test[0,:], test_Features.iloc[0,:], 
                    matplotlib=True, show=False, figsize=(20,10))
    plt.title('Test Set: SHAP Force Plot', y=1.75, fontsize=35)
    plt.tick_params(axis='x',labelsize=15)
    plt.tight_layout()
    plt.savefig('./results/XGB_ShapForce_TestSet.png', dpi=my_dpi*10, bbox_inches='tight');

    # Stop Logging
    mlflow.end_run()

if __name__ == '__main__':
    main()
