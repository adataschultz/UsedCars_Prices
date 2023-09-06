import os
import random
import numpy as np
import warnings
import argparse
import pandas as pd
import mlflow
import mlflow.xgboost
import joblib
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

seed_value = 42
os.environ['usedCars_xgbGPU'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def main():
    """Main function of the script."""

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to input train data")
    parser.add_argument("--test_data", type=str, help="path to input test data")
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--max_depth", required=False, default=6, type=int)
    parser.add_argument("--subsample", required=False, default=1, type=float)
    parser.add_argument("--gamma", required=False, default=0, type=float)
    parser.add_argument("--learning_rate", required=False, default=0.3, type=float)
    parser.add_argument("--reg_alpha", required=False, default=0, type=float)
    parser.add_argument("--reg_lambda", required=False, default=1, type=float)
    parser.add_argument("--colsample_bytree", required=False, default=1, type=float)
    parser.add_argument("--colsample_bylevel", required=False, default=1, type=float)
    parser.add_argument("--min_child_weight", required=False, default=1, type=int)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.xgboost.autolog()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input train data:", args.train_data)
    print("input test data:", args.test_data)
    
    trainDF = pd.read_csv(args.train_data, low_memory=False)
    testDF = pd.read_csv(args.test_data, low_memory=False)

    train_label = trainDF[['price']]
    test_label = testDF[['price']]

    train_features = trainDF.drop(columns = ['price'])
    test_features = testDF.drop(columns = ['price'])

    train_features = pd.get_dummies(train_features, drop_first=True)
    test_features = pd.get_dummies(test_features, drop_first=True)

    mlflow.log_metric("num_samples", train_features.shape[0])
    mlflow.log_metric("num_features", train_features.shape[1])

    print(f"Training with data of shape {train_features.shape}")

    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    best_model = XGBRegressor(objective='reg:squarederror',
                              metric='rmse',
                              booster='gbtree',
                              tree_method='gpu_hist',
                              scale_pos_weight=1,
                              use_label_encoder=False,
                              random_state=42,
                              verbosity=0,
                              n_estimators=args.n_estimators,
						      max_depth=args.max_depth,
						      subsample=args.subsample,
						      gamma=args.gamma,
						      learning_rate=args.learning_rate,
						      reg_alpha=args.reg_alpha,
						      reg_lambda=args.reg_lambda,
						      colsample_bytree=args.colsample_bytree,
						      colsample_bylevel=args.colsample_bylevel,
						      min_child_weight=args.min_child_weight)

    best_model.fit(train_features, train_label)

    print('\nModel Metrics for Used Cars XGBoost')
    y_train_pred = best_model.predict(train_features)
    y_test_pred = best_model.predict(test_features)

    train_mae = mean_absolute_error(train_label, y_train_pred)
    test_mae = mean_absolute_error(test_label, y_test_pred)
    train_mse = mean_squared_error(train_label, y_train_pred)
    test_mse = mean_squared_error(test_label, y_test_pred)
    train_rmse = mean_squared_error(train_label, y_train_pred, squared=False)
    test_rmse = mean_squared_error(test_label, y_test_pred, squared=False)
    train_r2 = r2_score(train_label, y_train_pred)
    test_r2 = r2_score(test_label, y_test_pred)

    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_r2", test_r2)

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
    print("Registering the model via MLFlow")
    mlflow.xgboost.log_model(
        xgb_model=best_model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.xgboost.save_model(
        xgb_model=best_model,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
    ###########################
    #</save and register model>
    ###########################
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
