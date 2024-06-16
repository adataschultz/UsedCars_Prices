# Prediction of the Price of Used Vehicles with `Streamlit`-`FastAPI` Model Serving

The structure of deploying models with a `Streamlit` frontend and `FastAPI` backend is as follows:
```
 |-README.md
 |-docker-compose.yml
 |-backend
 | |-Dockerfile
 | |-requirements.txt
 | |-main.py
 | |-xgboost
 | | |-model
 | | | |-usedcars_xgb_model.bin
 | |-lightgbm
 | | |-model
 | | | |-usedcars_lgbm_model.pkl
 | |-catboost
 | | |-model
 | | | |-catboost_training.json
 | | | |-test
 | | | | |-events.out.tfevents
 | | | |-usedcars_cat_model
 | | | |-test_error.tsv
 | | | |-learn_error.tsv
 | | | |-time_left.tsv
 | | | |-tmp
 | | | |-learn
 | | | | |-events.out.tfevents
 |-frontend
 | |-Dockerfile
 | |-requirements.txt
 | |-app.py
 | |-data
 | | |-usedCars_testSet.csv
 | | |-usedCars_trainSet.csv
 | |-static
 | | |-data_quality_dataset_report_columnLevel.html
 | | |-data_drift_column_report_backlegroom.html
 | | |-data_drift_report.html
 | | |-data_qualityTestPresets_report.html
 | | |-data_integrity_dataset_report_PriceListedData.html
 | | |-DataDriftTable_kl_report.html
 | | |-data_stabilityTestPresets_report.html
 | | |-data_quality_dataset_report_DatasetCorrelationsMetric.html
 | | |-DataQualityPreset_report.html
 | | |-data_driftTestPresets_report_stattest_psi.html
 | | |-data_integrity_dataset_report_SummaryMissing.html
 | | |-no_target_performance_report_horsepower_mileage.html
 | | |-train_domColorState.html
 | | |-test_domColorState.html   
 | | |-train_monthPriceState.html
 | | |-test_monthPriceState.html  
 | | |-train_priceColor.html
 | | |-test_priceColor.html    
 | | |-train_priceColorState.html
 | | |-test_priceColorState.html  
 | | |-traintest_priceState.html
 | | |-train_yearState.html
 | | |-test_yearState.html
 | |-xgboost
 | | |-train_model_booster.py
 | | |-model
 | | | |-usedcars_xgb_model.bin
 | | |-results
 | | | |-XGB_FeatureImportance.png
 | | | |-XGB_ShapSummary_TestSet.png
 | | | |-XGB_ShapForce_TrainSet.png
 | | | |-XGB_ShapSummary_TrainSet.png
 | | | |-XGB_ShapForce_TestSet.png
 | |-lightgbm
 | | |-model
 | | | |-usedcars_lgbm_model.pkl
 | | |-results
 | | | |-LGBM_ShapSummary_TestSet.png
 | | | |-LGBM_ShapSummary_TrainSet.png
 | | | |-LGBM_ShapForce_TestSet.png
 | | | |-LGBM_ShapForce_TrainSet.png
 | | | |-LGBM_FeatureImportance.png
 | | |-train_model.py
 | |-catboost
 | | |-model
 | | | |-catboost_training.json
 | | | |-test
 | | | | |-events.out.tfevents
 | | | |-usedcars_cat_model
 | | | |-test_error.tsv
 | | | |-learn_error.tsv
 | | | |-time_left.tsv
 | | | |-tmp
 | | | |-learn
 | | | | |-events.out.tfevents
 | | |-results
 | | | |-Cat_ShapSummary_TestSet.png
 | | | |-Cat_ShapSummary_TrainSet.png
 | | | |-Cat_FeatureImportance.png
 | | | |-Cat_ShapForce_TestSet.png
 | | | |-Cat_ShapForce_TrainSet.png
 | | |-train_model.py
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