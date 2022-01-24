# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
######################   UsedCarPrices_CarGurus  ##############################
######################  Regression - Nonlinear   ##############################
######################     light GBM Methods     ##############################
###############################################################################
import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from joblib import parallel_backend
from hyperopt import STATUS_OK
import lightgbm as lgb
from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin, tpe, Trials
import csv
from timeit import default_timer as timer
import ast
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import eli5 as eli
from eli5.sklearn import PermutationImportance 
from eli5 import show_weights
import webbrowser
from eli5.sklearn import explain_weights_sklearn
from eli5.formatters import format_as_dataframe, format_as_dataframes
from eli5 import show_prediction
import lime
from lime import lime_tabular

pd.set_option('display.max_columns', None)

# Set seed 
seed_value = 42
os.environ['UsedCarPrices_CarGurus_lightGBM'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Set path
path = r'D:\UsedCarPrices_CarGurus\Data'
os.chdir(path)

# Read data
df = pd.read_csv('usedCars_final.csv', low_memory=False)
df = df.drop_duplicates()
print('\nDimensions of Final Used Car Data:', df.shape) 
print('======================================================================') 

# Prepare for partitioning data
X = df.drop(['price'],axis=1)
y = df['price']

# Set up train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = seed_value)

# Train: Create dummy variables for categorical variables
X_train = pd.get_dummies(X_train, drop_first=True)

# Test: Create dummy variables for categorical variables
X_test = pd.get_dummies(X_test, drop_first=True)

# Set path for ML results
path = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM'
os.chdir(path)

###############################################################################
############################# light GBM HPO  ##################################
#############################   100 trials   ##################################
###############################################################################
# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X_train, label = y_train, params=params)

# Define an objective function

NUM_EVAL = 100
N_FOLDS = 3

def objective(params, n_folds = N_FOLDS):
    """lightGBM HPO"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for param_name in ['max_depth', 'num_leaves']:
        params[param_name] = int(params[param_name])
        
    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 1000, 
                        nfold = N_FOLDS,  stratified=False, 
                        early_stopping_rounds = 10, metrics = 'rmse', 
                        seed = seed_value)
    
    run_time = timer() - start
    
    loss = cv_results['rmse-mean'][-1]
        
    # Boosting rounds that returned the lowest cv score
    n_estimators = int(np.argmin(cv_results['rmse-mean']) + 1)
    
    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 'train_time': run_time,
            'status': STATUS_OK}

# Define the parameter grid
param_grid = {
    'force_col_wise': hp.choice('force_col_wise', "+"),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': hp.choice('max_depth', np.arange(5, 6, dtype=int)),
    'num_leaves': hp.choice('num_leaves', np.arange(30, 100, dtype=int)),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

# Optimization algorithm
tpe_algorithm = tpe.suggest

# File to save first results
out_file = 'lightGBM_UsedCars_CarGurus_trials_100.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration',  'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_trials,
                  rstate= np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) first
bayesOpt_trials_results = sorted(bayesOpt_trials.results, 
                                 key = lambda x: x['loss'])
print('Top two trials with the lowest loss (lowest RMSE)')
print(bayesOpt_trials_results[:2])

# Access results
results = pd.read_csv('lightGBM_UsedCars_CarGurus_trials_100.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMRegressor(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'regression',
                                      random_state = seed_value, 
                                      **best_bayes_params)

# Fit the model
best_bayes_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_UsedCars_CarGurus.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_UsedCars_CarGurus.pkl')
# print(model)
# =============================================================================
    
# Model Metrics
y_train_pred = best_bayes_model.predict(X_train)
y_test_pred = best_bayes_model.predict(X_test)

print('\nModel Metrics for lightGBM HPO UsedCars_CarGurus')
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# Evaluate on the testing data 
print('The best model from Bayes optimization scores {:.5f} MSE on the test set.'.format(mean_squared_error(y_test, y_test_pred)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Save dataframes of parameters
bayes_params.to_csv('bayes_paramslightGBM_HPO_UsedCars_CarGurus_100.csv', 
                    index = False)

# Convert data types for graphing
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', 
            linewidth = 2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise',
                     'max_depth']:
        plt.figure(figsize = (14, 6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label = 'Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0,
                                                                      'dart': 1,
                                                                      'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
i = 0
for i, hpo in enumerate(['learning_rate', 'num_leaves', 'colsample_bytree']):
        # Scatterplot
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize = (14, 6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

X_test1 = pd.DataFrame(X_test, columns=X_test.columns)                                                                    

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_100_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_100_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('UsedCarPrices_CarGurus_best_bayes_UsedCars_CarGurus_100_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_test1.iloc[[1]], 
                            feature_names = X_test1.columns.tolist(),
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_100_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_100_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_test1),
    feature_names=X_test1.columns,
    class_names=['price'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X_test1.iloc[1], 
    predict_fn=best_bayes_model.predict)

exp.save_to_file('UsedCarPrices_CarGurus_lightGBM_HPO_LIME.html')

###############################################################################
############################# light GBM HPO  ##################################
#############################   300 trials   ##################################
###############################################################################
# Change number of trials
NUM_EVAL = 300

# File to save first results
out_file = 'lightGBM_UsedCars_CarGurus_trials_300.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration',  'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_trials,
                  rstate= np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) first
bayesOpt_trials_results = sorted(bayesOpt_trials.results, 
                                 key = lambda x: x['loss'])
print('Top two trials with the lowest loss (lowest RMSE)')
print(bayesOpt_trials_results[:2])

# Access results
results = pd.read_csv('lightGBM_UsedCars_CarGurus_trials_300.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMRegressor(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'regression',
                                      random_state = seed_value, 
                                      **best_bayes_params)

# Fit the model
best_bayes_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_UsedCars_CarGurus_300.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_UsedCars_CarGurus)_300.pkl')
# print(model)
# =============================================================================
    
# Model Metrics
y_train_pred = best_bayes_model.predict(X_train)
y_test_pred = best_bayes_model.predict(X_test)

print('\nModel Metrics for lightGBM HPO UsedCars_CarGurus 300 trials')
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# Evaluate on the testing data 
print('The best model from Bayes optimization scores {:.5f} MSE on the test set.'.format(mean_squared_error(y_test, y_test_pred)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Save dataframes of parameters
bayes_params.to_csv('bayes_paramslightGBM_HPO_UsedCars_CarGurus_300.csv', 
                    index = False)

# Convert data types for graphing
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', 
            linewidth = 2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise',
                     'max_depth']:
        plt.figure(figsize = (14, 6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label = 'Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0,
                                                                      'dart': 1,
                                                                      'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
i = 0
for i, hpo in enumerate(['learning_rate', 'num_leaves', 'colsample_bytree']):
        # Scatterplot
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize = (14, 6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

X_test1 = pd.DataFrame(X_test, columns=X_test.columns)                                                                    

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_300_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('UsedCarPrices_CarGurus_best_bayes_UsedCars_CarGurus_300_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_test1.iloc[[1]], 
                            feature_names = X_test1.columns.tolist(),
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_300_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_300_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_test1),
    feature_names=X_test1.columns,
    class_names=['price'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X_test1.iloc[1], 
    predict_fn=best_bayes_model.predict)

exp.save_to_file('UsedCarPrices_CarGurus_lightGBM_HPO_300trials_LIME.html')

###############################################################################
############################# light GBM HPO  ##################################
#############################   500 trials   ##################################
###############################################################################
# Change number of trials
NUM_EVAL = 500

# File to save first results
out_file = 'lightGBM_UsedCars_CarGurus_trials_500.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration',  'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_trials,
                  rstate= np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) first
bayesOpt_trials_results = sorted(bayesOpt_trials.results, 
                                 key = lambda x: x['loss'])
print('Top two trials with the lowest loss (lowest RMSE)')
print(bayesOpt_trials_results[:2])

# Access results
results = pd.read_csv('lightGBM_UsedCars_CarGurus_trials_500.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMRegressor(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'regression',
                                      random_state = seed_value, 
                                      **best_bayes_params)

# Fit the model
best_bayes_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_UsedCars_CarGurus_500.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_UsedCars_CarGurus)_500.pkl')
# print(model)
# =============================================================================
    
# Model Metrics
y_train_pred = best_bayes_model.predict(X_train)
y_test_pred = best_bayes_model.predict(X_test)

print('\nModel Metrics for lightGBM HPO UsedCars_CarGurus 500 trials')
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# Evaluate on the testing data 
print('The best model from Bayes optimization scores {:.5f} MSE on the test set.'.format(mean_squared_error(y_test, y_test_pred)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Save dataframes of parameters
bayes_params.to_csv('bayes_paramslightGBM_HPO_UsedCars_CarGurus_500.csv', 
                    index = False)

# Convert data types for graphing
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', 
            linewidth = 2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise',
                     'max_depth']:
        plt.figure(figsize = (14, 6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label = 'Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0,
                                                                      'dart': 1,
                                                                      'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
i = 0
for i, hpo in enumerate(['learning_rate', 'num_leaves', 'colsample_bytree']):
        # Scatterplot
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize = (14, 6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

X_test1 = pd.DataFrame(X_test, columns=X_test.columns)                                                                    

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_500_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_500_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('UsedCarPrices_CarGurus_best_bayes_UsedCars_CarGurus_500_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_test1.iloc[[1]], 
                            feature_names = X_test1.columns.tolist(),
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_500_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_500_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_test1),
    feature_names=X_test1.columns,
    class_names=['price'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X_test1.iloc[1], 
    predict_fn=best_bayes_model.predict)

exp.save_to_file('UsedCarPrices_CarGurus_lightGBM_HPO_500trials_LIME.html')

###############################################################################
############################# light GBM HPO  ##################################
#############################  1000 trials   ##################################
###############################################################################
# Change number of trials
NUM_EVAL = 1000

# File to save first results
out_file = 'lightGBM_UsedCars_CarGurus_trials_1000.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration',  'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_trials,
                  rstate= np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) first
bayesOpt_trials_results = sorted(bayesOpt_trials.results, 
                                 key = lambda x: x['loss'])
print('Top two trials with the lowest loss (lowest RMSE)')
print(bayesOpt_trials_results[:2])

# Access results
results = pd.read_csv('lightGBM_UsedCars_CarGurus_trials_1000.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMRegressor(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'regression',
                                      random_state = seed_value, 
                                      **best_bayes_params)

# Fit the model
best_bayes_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_UsedCars_CarGurus_1000.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_UsedCars_CarGurus)_1000.pkl')
# print(model)
# =============================================================================
    
# Model Metrics
y_train_pred = best_bayes_model.predict(X_train)
y_test_pred = best_bayes_model.predict(X_test)

print('\nModel Metrics for lightGBM HPO UsedCars_CarGurus 1000 trials')
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# Evaluate on the testing data 
print('The best model from Bayes optimization scores {:.5f} MSE on the test set.'.format(mean_squared_error(y_test, y_test_pred)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Save dataframes of parameters
bayes_params.to_csv('bayes_paramslightGBM_HPO_UsedCars_CarGurus_1000.csv', 
                    index = False)

# Convert data types for graphing
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', 
            linewidth = 2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise',
                     'max_depth']:
        plt.figure(figsize = (14, 6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label = 'Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0,
                                                                      'dart': 1,
                                                                      'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
i = 0
for i, hpo in enumerate(['learning_rate', 'num_leaves', 'colsample_bytree']):
        # Scatterplot
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize = (14, 6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

X_test1 = pd.DataFrame(X_test, columns=X_test.columns)                                                                    

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_1000_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_1000_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('UsedCarPrices_CarGurus_best_bayes_UsedCars_CarGurus_1000_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_test1.iloc[[1]], 
                            feature_names = X_test1.columns.tolist(),
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_1000_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_1000_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_test1),
    feature_names=X_test1.columns,
    class_names=['price'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X_test1.iloc[1], 
    predict_fn=best_bayes_model.predict)

exp.save_to_file('UsedCarPrices_CarGurus_lightGBM_HPO_1000trials_LIME.html')

###############################################################################
############################# light GBM GBDT HPO  #############################
#############################     100 trials      #############################
###############################################################################
# Change number of trials
NUM_EVAL = 100

# Define the search space for GBDT
param_grid = {
    'force_col_wise': hp.choice('force_col_wise', "+"),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(1)),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 0.8),
    'min_sum_hessian_in_leaf': hp.choice('min_sum_hessian_in_leaf',  np.arange(0.1, 1, dtype=int)),
    'max_depth': hp.choice('max_depth', np.arange(3, 15, dtype=int)),
    'num_leaves': hp.choice('num_leaves', np.arange(30, 200, dtype=int)),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.3, 1)}]),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

# File to save first results
out_file = 'lightGBM_UsedCars_CarGurus_trials_gbdt_100.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration',  'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_trials,
                  rstate= np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) first
bayesOpt_trials_results = sorted(bayesOpt_trials.results, key = lambda x: x['loss'])
print('Top two trials with the lowest loss (lowest MAE)')
print(bayesOpt_trials_results[:2])

# Access results
results = pd.read_csv('lightGBM_UsedCars_CarGurus_trials_gbdt_100.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMRegressor(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'regression',
                                      random_state = seed_value, **best_bayes_params)

# Fit the model
best_bayes_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_UsedCars_CarGurus_gbdt_100.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_UsedCars_CarGurus_gbdt_100.pkl')
# print(model)
# =============================================================================
    
# Model Metrics
y_train_pred = best_bayes_model.predict(X_train)
y_test_pred = best_bayes_model.predict(X_test)

print('\nModel Metrics for lightGBM HPO UsedCars_CarGurus GBDT 100 trials')
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# Evaluate on the testing data 
print('The best model from Bayes optimization scores {:.5f} MSE on the test set.'.format(mean_squared_error(y_test, y_test_pred)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Save dataframes of parameters
bayes_params.to_csv('bayes_paramslightGBM_HPO_UsedCars_CarGurus_gbdt_100.csv', 
                    index = False)

# Convert data types for graphing
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['feature_fraction'] = bayes_params['feature_fraction'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', 
            linewidth = 2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise',
                     'max_depth', 'min_sum_hessian_in_leaf']:
        plt.figure(figsize = (14, 6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label = 'Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'gbdt': 0})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0], ['gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
i = 0
for i, hpo in enumerate(['learning_rate', 'num_leaves', 'feature_fraction']):
        # Scatterplot
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize = (14, 6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

X_test1 = pd.DataFrame(X_test, columns=X_test.columns)                                                                    

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_GBDT_100_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_GBDT_100_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('UsedCarPrices_CarGurus_best_bayes_UsedCars_CarGurus_GBDT_100_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_test1.iloc[[1]], 
                            feature_names = X_test1.columns.tolist(),
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_GBDT_100_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_GBDT_100_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_test1),
    feature_names=X_test1.columns,
    class_names=['price'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X_test1.iloc[1], 
    predict_fn=best_bayes_model.predict)

exp.save_to_file('UsedCarPrices_CarGurus_lightGBM_HPO_GBDT_100_LIME.html')

###############################################################################
############################# light GBM GBDT HPO  #############################
#############################     500 trials      #############################
###############################################################################
# Change number of trials
NUM_EVAL = 500

# File to save first results
out_file = 'lightGBM_UsedCars_CarGurus_trials_gbdt_500.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration',  'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_trials,
                  rstate= np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) first
bayesOpt_trials_results = sorted(bayesOpt_trials.results, key = lambda x: x['loss'])
print('Top two trials with the lowest loss (lowest MAE)')
print(bayesOpt_trials_results[:2])

# Access results
results = pd.read_csv('lightGBM_UsedCars_CarGurus_trials_gbdt_500.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMRegressor(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'regression',
                                      random_state = seed_value, **best_bayes_params)

# Fit the model
best_bayes_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_UsedCars_CarGurus_gbdt_500.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_UsedCars_CarGurus_gbdt_500.pkl')
# print(model)
# =============================================================================
    
# Model Metrics
y_train_pred = best_bayes_model.predict(X_train)
y_test_pred = best_bayes_model.predict(X_test)

print('\nModel Metrics for lightGBM HPO UsedCars_CarGurus GBDT 500 trials')
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# Evaluate on the testing data 
print('The best model from Bayes optimization scores {:.5f} MSE on the test set.'.format(mean_squared_error(y_test, y_test_pred)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Save dataframes of parameters
bayes_params.to_csv('bayes_paramslightGBM_HPO_UsedCars_CarGurus_gbdt_500.csv', 
                    index = False)

# Convert data types for graphing
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['feature_fraction'] = bayes_params['feature_fraction'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', 
            linewidth = 2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise',
                     'max_depth', 'min_sum_hessian_in_leaf']:
        plt.figure(figsize = (14, 6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label = 'Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'gbdt': 0})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0], ['gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
i = 0
for i, hpo in enumerate(['learning_rate', 'num_leaves', 'feature_fraction']):
        # Scatterplot
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize = (14, 6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), 
                   title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

X_test1 = pd.DataFrame(X_test, columns=X_test.columns)                                                                    

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_GBDT_500_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_GBDT_500_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('UsedCarPrices_CarGurus_best_bayes_UsedCars_CarGurus_GBDT_500_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_test1.iloc[[1]], 
                            feature_names = X_test1.columns.tolist(),
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_GBDT_500_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\UsedCarPrices_CarGurus\ML\ML_Results\NonLinear\lightGBM\best_bayes_UsedCars_CarGurus_GBDT_500_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_test1),
    feature_names=X_test1.columns,
    class_names=['price'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X_test1.iloc[1], 
    predict_fn=best_bayes_model.predict)

exp.save_to_file('UsedCarPrices_CarGurus_lightGBM_HPO_GBDT_500_LIME.html')

###############################################################################