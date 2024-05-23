import os
import random
import numpy as np
import warnings
import pandas as pd
import mlflow
from catboost import CatBoostRegressor
import joblib
import dill as pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap 
warnings.filterwarnings('ignore')
my_dpi = 96

seed_value = 42
os.environ['usedCars_catGPU'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def main():
    '''Main function of the script.'''

    # Start Logging
    with mlflow.start_run():
    
        ###################
        #<prepare the data>
        ###################
        train_path = '../data/usedCars_trainSet.csv'
        trainDF = pd.read_csv(train_path, low_memory=False)
        #trainDF  = trainDF.sample(frac=0.4, random_state=42)

        test_path = '../data/usedCars_testSet.csv'
        testDF = pd.read_csv(test_path, low_memory=False)
        #testDF = testDF.sample(frac=0.4, random_state=42)

        train_label = trainDF[['price']]
        test_label = testDF[['price']]

        train_features = trainDF.drop(['price'], axis=1)
        test_features = testDF.drop(['price'], axis=1)

        train_features['is_new'] = train_features['is_new'].astype(str)
        test_features['is_new'] = test_features['is_new'].astype(str) 
        
        categorical_features_indices = ['body_type', 'fuel_type', 'listing_color',
                                        'transmission', 'wheel_system_display', 'State',
                                        'listed_date_yearMonth', 'is_new']

        mlflow.log_metric('num_samples', train_features.shape[0])
        mlflow.log_metric('num_features', train_features.shape[1])

        print(f'Training with data of shape {train_features.shape}')

        ####################
        #</prepare the data>
        ####################

        ##################
        #<train the model>
        ##################
        param = {
            'loss_function': 'RMSE',
            'task_type': 'GPU',
            'early_stopping_rounds': 10,
            'rsm': 1,
            'logging_level': 'Silent',
            'n_estimators': 461,
            'learning_rate': 0.17753236478366918,
            'depth': 10,
            'l2_leaf_reg': 0.41721695797669994,
            'min_data_in_leaf': 11,
            'one_hot_max_size': 13,
            'random_state': 42,
            'train_dir':'./model/'}
        
        mlflow.set_tag('usedcars_model', 'catboost')
        mlflow.set_tag('data scientist', 'ars')
        mlflow.log_params(param)

        usedcars_cat_model = CatBoostRegressor(**param)

        usedcars_cat_model.fit(train_features, train_label, eval_set=[(test_features, test_label)], cat_features=categorical_features_indices)#, verbose=0)

        print('\nModel Metrics for Used Cars Catboost')
        y_train_pred = usedcars_cat_model.predict(train_features)
        y_test_pred = usedcars_cat_model.predict(test_features)

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
        mlflow.catboost.log_model(
            cb_model=usedcars_cat_model,
            registered_model_name='usedcars_cat_model',
            artifact_path='usedcars_cat_model',
        )

        # Saving the model to a file
        model_dir = './model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        usedcars_cat_model.save_model('./model/usedcars_cat_model')  

        ###########################
        #</save and register model>
        ###########################
        # Feature Importance
        def plot_feature_importance(importance, names, model_type): 
            feature_importance = np.array(importance)
            feature_names = np.array(names)
            data={'feature_names': feature_names,
                  'feature_importance': feature_importance}
            fi_df = pd.DataFrame(data)
            fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
            fig = plt.figure(figsize=(15,10))
            sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'],
                        palette='cool')
            plt.title(model_type + ' Feature Importance')
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature Names')
            plt.tight_layout()
            plt.savefig('./results/Cat_FeatureImportance.png', dpi=my_dpi*10, bbox_inches='tight');

        plot_feature_importance(usedcars_cat_model.get_feature_importance(),
                                train_features.columns, 'Catboost')
        #plt.tight_layout()
        #plt.savefig('./results/Cat_FeatureImportance.png', dpi=my_dpi*10, bbox_inches='tight');
        
        # SHAP   
        shap_explainer = shap.TreeExplainer(usedcars_cat_model)
    
        ex_filename = './model/Cat_ShapExplainer.pkl'
        joblib.dump(shap_explainer, filename=ex_filename, compress=('bz2', 9))

        shap_values_train = shap_explainer.shap_values(train_features, train_label, check_additivity=False)

        sv_filename = './model/Cat_shap_values_train.pkl'

        joblib.dump(shap_values_train, filename=sv_filename, compress=('bz2', 9))

        #ex_filename = './model/Cat_ShapExplainer.pkl'
        #shap_explainer = joblib.load(filename=ex_filename)
        
        #shap_val_filename = './model/Cat_Shap_values_train.pkl'
        #shap_values_train = joblib.load(filename=shap_val_filename)

        plt.rcParams.update({'font.size': 5})
        fig = plt.figure(figsize=(5,10))
        shap.summary_plot(shap_values_train, train_features, show=False)
        plt.title('Train Set: SHAP Summary Plot', y=1.1, fontsize=14)
        plt.tight_layout()
        plt.savefig('./results/Cat_ShapSummary_TrainSet.png', dpi=my_dpi*10, bbox_inches='tight');

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figsize=(20,20))
        shap.force_plot(shap_explainer.expected_value, shap_values_train[0,:], train_features.iloc[0,:], matplotlib=True, show=False, figsize=(20, 10))#, unsafe_allow_html=True)
        plt.title('Train Set: SHAP Force Plot', y=1.75, fontsize=35)
        plt.tick_params(labelsize=25)
        plt.tight_layout()
        plt.savefig('./results/Cat_ShapForce_TrainSet.png', dpi=my_dpi*10, bbox_inches='tight');

        # Test set
        shap_values_test = shap_explainer.shap_values(test_features, test_label, check_additivity=False)
    
        sv_filename = './model/Cat_shap_values_test.pkl'
        joblib.dump(shap_values_test, filename=sv_filename, compress=('bz2', 9))
        
        #shap_val_filename = './model/Cat_Shap_values_test.pkl'
        #shap_values_test = joblib.load(filename=shap_val_filename)

        plt.rcParams.update({'font.size': 5})
        fig = plt.figure(figsize=(5,10))
        shap.summary_plot(shap_values_test, test_features, show=False)
        plt.title('Test Set: SHAP Summary Plot', y=1.1, fontsize=14)
        plt.tight_layout()
        plt.savefig('./results/Cat_ShapSummary_TestSet.png', dpi=my_dpi*10, bbox_inches='tight');

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figsize=(20,20))
        shap.force_plot(shap_explainer.expected_value, shap_values_test[0,:], test_features.iloc[0,:], matplotlib=True, show=False, figsize=(20, 10))
        plt.title('Test Set: SHAP Force Plot', y=1.75, fontsize=35)
        plt.tick_params(labelsize=25)
        plt.tight_layout()
        plt.savefig('./results/Cat_ShapForce_TestSet.png', dpi=my_dpi*10, bbox_inches='tight');
        
        # Stop Logging
        mlflow.end_run()

if __name__ == '__main__':
    main()
