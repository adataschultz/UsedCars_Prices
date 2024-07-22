# Prediction of the Price of Used Vehicles with `Streamlit`-`FastAPI` Model Serving

The structure of deploying models with a `Streamlit` frontend and `FastAPI` backend is as follows:
```
├── backend
│   ├── catboost
│   │   └── model
│   │       ├── catboost_training.json
│   │       ├── learn
│   │       │   └── events.out.tfevents
│   │       ├── learn_error.tsv
│   │       ├── test
│   │       │   └── events.out.tfevents
│   │       ├── test_error.tsv
│   │       ├── time_left.tsv
│   │       └── usedcars_cat_model
│   ├── Dockerfile
│   ├── lightgbm
│   │   └── model
│   │       └── usedcars_lgbm_model.pkl
│   ├── main.py
│   ├── requirements.txt
│   └── xgboost
│       └── model
│           └── usedcars_xgb_model.bin
├── docker-compose.yml
├── frontend
│   ├── app.py
│   ├── catboost
│   │   ├── mlruns
│   │   │   ├── 0
│   │   │   │   ├── 5af33b60af8040cea7abc3ff7f3a9462
│   │   │   │   │   ├── artifacts
│   │   │   │   │   │   └── usedcars_cat_model
│   │   │   │   │   │       ├── conda.yaml
│   │   │   │   │   │       ├── metadata
│   │   │   │   │   │       │   ├── conda.yaml
│   │   │   │   │   │       │   ├── MLmodel
│   │   │   │   │   │       │   ├── python_env.yaml
│   │   │   │   │   │       │   └── requirements.txt
│   │   │   │   │   │       ├── MLmodel
│   │   │   │   │   │       ├── model.cb
│   │   │   │   │   │       ├── python_env.yaml
│   │   │   │   │   │       └── requirements.txt
│   │   │   │   │   ├── meta.yaml
│   │   │   │   │   ├── metrics
│   │   │   │   │   │   ├── num_features
│   │   │   │   │   │   ├── num_samples
│   │   │   │   │   │   ├── test_mae
│   │   │   │   │   │   ├── test_mse
│   │   │   │   │   │   ├── test_r2
│   │   │   │   │   │   ├── test_rmse
│   │   │   │   │   │   ├── train_mae
│   │   │   │   │   │   ├── train_mse
│   │   │   │   │   │   ├── train_r2
│   │   │   │   │   │   └── train_rmse
│   │   │   │   │   ├── params
│   │   │   │   │   │   ├── depth
│   │   │   │   │   │   ├── early_stopping_rounds
│   │   │   │   │   │   ├── l2_leaf_reg
│   │   │   │   │   │   ├── learning_rate
│   │   │   │   │   │   ├── logging_level
│   │   │   │   │   │   ├── loss_function
│   │   │   │   │   │   ├── min_data_in_leaf
│   │   │   │   │   │   ├── n_estimators
│   │   │   │   │   │   ├── one_hot_max_size
│   │   │   │   │   │   ├── random_state
│   │   │   │   │   │   ├── rsm
│   │   │   │   │   │   ├── task_type
│   │   │   │   │   │   └── train_dir
│   │   │   │   │   └── tags
│   │   │   │   │       ├── data scientist
│   │   │   │   │       ├── mlflow.log-model.history
│   │   │   │   │       ├── mlflow.runName
│   │   │   │   │       ├── mlflow.source.git.commit
│   │   │   │   │       ├── mlflow.source.name
│   │   │   │   │       ├── mlflow.source.type
│   │   │   │   │       ├── mlflow.user
│   │   │   │   │       └── usedcars_model
│   │   │   │   └── meta.yaml
│   │   │   └── models
│   │   │       └── usedcars_cat_model
│   │   │           ├── meta.yaml
│   │   │           └── version-1
│   │   │               └── meta.yaml
│   │   ├── model
│   │   │   ├── catboost_training.json
│   │   │   ├── learn
│   │   │   │   └── events.out.tfevents
│   │   │   ├── learn_error.tsv
│   │   │   ├── test
│   │   │   │   └── events.out.tfevents
│   │   │   ├── test_error.tsv
│   │   │   ├── time_left.tsv
│   │   │   └── usedcars_cat_model
│   │   ├── results
│   │   │   ├── Cat_FeatureImportance.png
│   │   │   ├── Cat_ShapForce_TestSet.png
│   │   │   ├── Cat_ShapForce_TrainSet.png
│   │   │   ├── Cat_ShapSummary_TestSet.png
│   │   │   └── Cat_ShapSummary_TrainSet.png
│   │   └── train_model.py
│   ├── data
│   │   ├── usedCars_evalSet.csv
│   │   ├── usedCars_testSet.parquet.gzip
│   │   └── usedCars_trainSet.parquet.gzip
│   ├── Dockerfile
│   ├── lightgbm
│   │   ├── mlruns
│   │   │   ├── 0
│   │   │   │   ├── 129fd4cfb05a49229d04c515bcf35e23
│   │   │   │   │   ├── artifacts
│   │   │   │   │   │   ├── feature_importance_gain.json
│   │   │   │   │   │   ├── feature_importance_gain.png
│   │   │   │   │   │   ├── feature_importance_split.json
│   │   │   │   │   │   ├── feature_importance_split.png
│   │   │   │   │   │   ├── model
│   │   │   │   │   │   │   ├── conda.yaml
│   │   │   │   │   │   │   ├── metadata
│   │   │   │   │   │   │   │   ├── conda.yaml
│   │   │   │   │   │   │   │   ├── MLmodel
│   │   │   │   │   │   │   │   ├── python_env.yaml
│   │   │   │   │   │   │   │   └── requirements.txt
│   │   │   │   │   │   │   ├── MLmodel
│   │   │   │   │   │   │   ├── model.pkl
│   │   │   │   │   │   │   ├── python_env.yaml
│   │   │   │   │   │   │   └── requirements.txt
│   │   │   │   │   │   └── usedcars_lgbm_model
│   │   │   │   │   │       ├── conda.yaml
│   │   │   │   │   │       ├── metadata
│   │   │   │   │   │       │   ├── conda.yaml
│   │   │   │   │   │       │   ├── MLmodel
│   │   │   │   │   │       │   ├── python_env.yaml
│   │   │   │   │   │       │   └── requirements.txt
│   │   │   │   │   │       ├── MLmodel
│   │   │   │   │   │       ├── model.pkl
│   │   │   │   │   │       ├── python_env.yaml
│   │   │   │   │   │       └── requirements.txt
│   │   │   │   │   ├── inputs
│   │   │   │   │   │   ├── 1d9892b9ad55b1c82b408b0effbe4e54
│   │   │   │   │   │   │   └── meta.yaml
│   │   │   │   │   │   └── cb2b26149b2bf2b666d8e58a9e1143d2
│   │   │   │   │   │       └── meta.yaml
│   │   │   │   │   ├── meta.yaml
│   │   │   │   │   ├── metrics
│   │   │   │   │   │   ├── num_features
│   │   │   │   │   │   ├── num_samples
│   │   │   │   │   │   ├── test_mae
│   │   │   │   │   │   ├── test_mse
│   │   │   │   │   │   ├── test_r2
│   │   │   │   │   │   ├── test_rmse
│   │   │   │   │   │   ├── train_mae
│   │   │   │   │   │   ├── train_mse
│   │   │   │   │   │   ├── train_r2
│   │   │   │   │   │   └── train_rmse
│   │   │   │   │   ├── params
│   │   │   │   │   │   ├── bagging_freq
│   │   │   │   │   │   ├── boosting_type
│   │   │   │   │   │   ├── colsample_bytree
│   │   │   │   │   │   ├── device
│   │   │   │   │   │   ├── gpu_device_id
│   │   │   │   │   │   ├── gpu_platform_id
│   │   │   │   │   │   ├── lambda_l1
│   │   │   │   │   │   ├── lambda_l2
│   │   │   │   │   │   ├── learning_rate
│   │   │   │   │   │   ├── max_depth
│   │   │   │   │   │   ├── metric
│   │   │   │   │   │   ├── min_child_samples
│   │   │   │   │   │   ├── min_child_weight
│   │   │   │   │   │   ├── min_split_gain
│   │   │   │   │   │   ├── n_estimators
│   │   │   │   │   │   ├── num_leaves
│   │   │   │   │   │   ├── random_state
│   │   │   │   │   │   ├── reg_alpha
│   │   │   │   │   │   ├── reg_lambda
│   │   │   │   │   │   ├── subsample
│   │   │   │   │   │   ├── subsample_for_bin
│   │   │   │   │   │   ├── subsample_freq
│   │   │   │   │   │   └── verbosity
│   │   │   │   │   └── tags
│   │   │   │   │       ├── data scientist
│   │   │   │   │       ├── mlflow.log-model.history
│   │   │   │   │       ├── mlflow.runName
│   │   │   │   │       ├── mlflow.source.git.commit
│   │   │   │   │       ├── mlflow.source.name
│   │   │   │   │       ├── mlflow.source.type
│   │   │   │   │       ├── mlflow.user
│   │   │   │   │       └── usedcars_model
│   │   │   │   ├── datasets
│   │   │   │   │   ├── 5e2e0b833aa9e4c106d3eafc62d476fd
│   │   │   │   │   │   └── meta.yaml
│   │   │   │   │   └── 9d1495e8966956b86995dcd6a6fd2bab
│   │   │   │   │       └── meta.yaml
│   │   │   │   └── meta.yaml
│   │   │   └── models
│   │   │       └── usedcars_lgbm_model
│   │   │           ├── meta.yaml
│   │   │           └── version-1
│   │   │               └── meta.yaml
│   │   ├── model
│   │   │   └── usedcars_lgbm_model.pkl
│   │   ├── results
│   │   │   ├── LGBM_FeatureImportance.png
│   │   │   ├── LGBM_ShapForce_TestSet.png
│   │   │   ├── LGBM_ShapForce_TrainSet.png
│   │   │   ├── LGBM_ShapSummary_TestSet.png
│   │   │   └── LGBM_ShapSummary_TrainSet.png
│   │   └── train_model.py
│   ├── requirements.txt
│   ├── static
│   │   ├── data_drift_column_report_backlegroom.html
│   │   ├── data_drift_report.html
│   │   ├── DataDriftTable_kl_report.html
│   │   ├── data_driftTestPresets_report_stattest_psi.html
│   │   ├── data_integrity_dataset_report_PriceListedData.html
│   │   ├── data_integrity_dataset_report_SummaryMissing.html
│   │   ├── data_quality_dataset_report_columnLevel.html
│   │   ├── data_quality_dataset_report_DatasetCorrelationsMetric.html
│   │   ├── DataQualityPreset_report.html
│   │   ├── data_qualityTestPresets_report.html
│   │   ├── data_stabilityTestPresets_report.html
│   │   ├── no_target_performance_report_horsepower_mileage.html
│   │   ├── test_domColorState.html
│   │   ├── test_monthPriceState.html
│   │   ├── test_priceColor.html
│   │   ├── test_priceColorState.html
│   │   ├── test_yearState.html
│   │   ├── train_domColorState.html
│   │   ├── train_monthPriceState.html
│   │   ├── train_priceColor.html
│   │   ├── train_priceColorState.html
│   │   ├── traintest_priceState.html
│   │   └── train_yearState.html
│   ├── streamlit_app_metrics.py
│   ├── streamlit_app.py
│   ├── streamlit_app_script.py
│   └── xgboost
│       ├── mlruns
│       │   ├── 0
│       │   │   ├── 5fd3dfacad3c4eb1b7e6a72789776923
│       │   │   │   ├── artifacts
│       │   │   │   │   ├── feature_importance_weight.json
│       │   │   │   │   ├── feature_importance_weight.png
│       │   │   │   │   ├── model
│       │   │   │   │   │   ├── conda.yaml
│       │   │   │   │   │   ├── metadata
│       │   │   │   │   │   │   ├── conda.yaml
│       │   │   │   │   │   │   ├── MLmodel
│       │   │   │   │   │   │   ├── python_env.yaml
│       │   │   │   │   │   │   └── requirements.txt
│       │   │   │   │   │   ├── MLmodel
│       │   │   │   │   │   ├── model.xgb
│       │   │   │   │   │   ├── python_env.yaml
│       │   │   │   │   │   └── requirements.txt
│       │   │   │   │   └── usedcars_xgb_model
│       │   │   │   │       ├── conda.yaml
│       │   │   │   │       ├── metadata
│       │   │   │   │       │   ├── conda.yaml
│       │   │   │   │       │   ├── MLmodel
│       │   │   │   │       │   ├── python_env.yaml
│       │   │   │   │       │   └── requirements.txt
│       │   │   │   │       ├── MLmodel
│       │   │   │   │       ├── model.xgb
│       │   │   │   │       ├── python_env.yaml
│       │   │   │   │       └── requirements.txt
│       │   │   │   ├── inputs
│       │   │   │   │   └── 748dd6869bccc05b486d571777ff6d6d
│       │   │   │   │       └── meta.yaml
│       │   │   │   ├── meta.yaml
│       │   │   │   ├── metrics
│       │   │   │   │   ├── num_features
│       │   │   │   │   ├── num_samples
│       │   │   │   │   ├── test_mae
│       │   │   │   │   ├── test_mse
│       │   │   │   │   ├── test_r2
│       │   │   │   │   ├── test_rmse
│       │   │   │   │   ├── train_mae
│       │   │   │   │   ├── train_mse
│       │   │   │   │   ├── train_r2
│       │   │   │   │   └── train_rmse
│       │   │   │   ├── params
│       │   │   │   │   ├── booster
│       │   │   │   │   ├── colsample_bylevel
│       │   │   │   │   ├── colsample_bytree
│       │   │   │   │   ├── custom_metric
│       │   │   │   │   ├── device
│       │   │   │   │   ├── early_stopping_rounds
│       │   │   │   │   ├── gamma
│       │   │   │   │   ├── learning_rate
│       │   │   │   │   ├── max_depth
│       │   │   │   │   ├── maximize
│       │   │   │   │   ├── metric
│       │   │   │   │   ├── min_child_weight
│       │   │   │   │   ├── n_estimators
│       │   │   │   │   ├── num_boost_round
│       │   │   │   │   ├── objective
│       │   │   │   │   ├── random_state
│       │   │   │   │   ├── reg_alpha
│       │   │   │   │   ├── reg_lambda
│       │   │   │   │   ├── scale_pos_weight
│       │   │   │   │   ├── subsample
│       │   │   │   │   ├── tree_method
│       │   │   │   │   ├── use_label_encoder
│       │   │   │   │   ├── verbose_eval
│       │   │   │   │   └── verbosity
│       │   │   │   └── tags
│       │   │   │       ├── mlflow.log-model.history
│       │   │   │       ├── mlflow.runName
│       │   │   │       ├── mlflow.source.git.commit
│       │   │   │       ├── mlflow.source.name
│       │   │   │       ├── mlflow.source.type
│       │   │   │       └── mlflow.user
│       │   │   ├── datasets
│       │   │   │   └── 5bb29f5a5bf849d94254701b4ad6b709
│       │   │   │       └── meta.yaml
│       │   │   └── meta.yaml
│       │   └── models
│       │       └── usedcars_xgb_model
│       │           ├── meta.yaml
│       │           └── version-1
│       │               └── meta.yaml
│       ├── model
│       │   └── usedcars_xgb_model.bin
│       ├── results
│       │   ├── XGB_FeatureImportance.png
│       │   ├── XGB_ShapForce_TestSet.png
│       │   ├── XGB_ShapForce_TrainSet.png
│       │   ├── XGB_ShapSummary_TestSet.png
│       │   └── XGB_ShapSummary_TrainSet.png
│       └── train_model_booster.py
```

## To run the app locally using `Docker`:
Edit the endpoint in `frontend/app.py` to http://localhost:80/predict or the `Docker` url given the build.
Open a terminal with `Docker` running, then run <code>docker compose build</code> and successively <code>docker compose up</code>.

To visit the FastAPI UI, visit http://localhost:80 and to visit the streamlit UI use http://localhost:8501.

## `Streamlit Cloud` and `Heroku`
The frontend app can be viewed on `Streamlit Cloud` at https://predicting-used-vehicles-prices.streamlit.app/.

To deploy frontend app on `Heroku`, first [download](https://devcenter.heroku.com/articles/heroku-cli) the installer and `Heroku CLI`. Login to `Heroku` and run `heroku create app-usedcars --buildpack heroku/python` to create the app add the `Python buildpack`. Then add the remote using `heroku git:remote -a app-usedcar`. Navigate to `Deploy/app_usedCars/Heroku/frontend`, and create a `.slugignore` containing the directories and files to ignore. Create a `runtime.txt` file specifying the `Python` version to use (`python-3.10.14`). Then create a `setup.sh` that creates a `~/.streamlit/` directory, add your email as credentials for `~/.streamlit/credentials.toml`, and then specify the information for the `server` and `port` for `~/.streamlit/config.toml`. Next, create a `Procfile` file that says to run the `setup.sh` shell script and `streamlit_app.py` `Python` script using `web: sh setup.sh && streamlit run streamlit_app.py`. Navigate back to the repository root directory and run `git subtree push --prefix Deploy/app_usedCars/Heroku/frontend heroku main`, which will deploy the `Streamlit` frontend to `Heroku`. This app requires at least the `Standard` `Dynos` due to the limited memory availability on `Heroku`.    

## App Deployment on `Google Cloud Platform`
This application is currently hosted on `Google Cloud Platform` using `Cloud Run` using two separate containers for the frontend and backend.

The steps to host the app on `Google Cloud Platform` are:

1. Create a `GCP` billing account.
2. Enable `Cloud Run` and `Artifact Registry` APIs.
3. Activate the `Cloud Shell`, authorize docker in a specified region and create a repository in `Artifact Registry`.
4. Then clone the [Used Car Repository](https://github.com/adataschultz/UsedCars_Prices/tree/main) in the `Cloud Shell`.
5. Next, move to the `Deploy/app_usedCars/GCP_CloudRun` directory and run `docker compose build`.
5. Then tag and push both the frontend and backend to `Artifact Registry`.
6. Next, create the two `Cloud Run` services for the `Streamlit` frontend and `FastAPI` backend. Navigate to the `gcp_cloudrun-vehicle-app-backend` container in `Artifact Registry` and select `Deploy to Cloud Run`. Select `Allow unauthenticated invocations` for `Authentication`, `CPU is only allocated during request processing` for `CPU allocation and pricing`, `Port 80` for `Container port`, `8 GB` memory, `6 CPU`, `Second generation` for `Execution environment`, `16 Maximum number of instances` with `Startup CPU boost`. Once created, this will generate an URL that is needed to be added to the `app.py` script in the section where an external data source can be uploaded for real time vehicle price prediction. Once this is added, the `frontend` container needs to be rebuilt, tagged and pushed to `Artifact Registry`. Then navigate to the `gcp_cloudrun-vehicle-app` frontend container and select `Deploy to Cloud Run`. Select `Allow unauthenticated invocations` for `Authentication`, `CPU is only allocated during request processing` for `CPU allocation and pricing`, `Port 8501` for `Container port`, `16 GB` memory, `8 CPU`, `Second generation` for `Execution environment`, `12 Maximum number of instances` with `Startup CPU boost`. Once created, this generated URL contains the `Streamlit` frontend that is able to make requests to the `FastAPI` backend for real time vehicle price predictions.
7. Connect `Cloud Build` to the `Git` repository to set up a trigger when the `main` branch is updated.