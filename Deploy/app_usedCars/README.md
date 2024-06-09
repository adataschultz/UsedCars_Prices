# Prediction of the Price of Used Vehicles with Streamlit-FastAPI Model Serving

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
 | | |-DataQualityPreset_
 | | |-DataQualityPreset_report.html
 | | |-data_driftTestPresets_report_stattest_psi.html
 | | |-data_integrity_dataset_report_SummaryMissing.html
 | | |-no_target_performance_report_horsepower_mileage.html
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

## To run the app locally using docker compose:
Edit the endpoint in `frontend/app.py` to http://localhost:80/predict or the docker url given the build.
Open a terminal with docker running, then run:<br>

<code>docker compose build</code><br>
and successively<br>

<code>docker compose up</code>

To visit the FastAPI UI, visit http://localhost:80 and to visit the streamlit UI use http://localhost:8501

# `Streamlit Cloud` and `Heroku`
The frontend app can be viewed on `Streamlit Cloud` at https://predicting-used-vehicles-prices.streamlit.app/

The frontend app can be viewed on `Heroku` at https://app-usedcars-836bbb4de9f1.herokuapp.com/

## App Deployment on Google Cloud Platform
This application is currently hosted on `Google Cloud Platform` using `Cloud Run` using two separate containers for the frontend and backend.

The steps to host the app on Google Cloud Platform are:

1. Create a `GCP` billing account.
2. Enable `Cloud Run` and `Artifact Registry` APIs.
3. Activate the `Cloud Shell`, authorize docker in a specified region and create a repository in `Artifact Registry`
4. Then clone the [Used Car Repository](https://github.com/adataschultz/UsedCars_Prices/tree/main) in the `Cloud Shell`
5. Next, move to the `Deploy/app_usedCars` directory and run `docker compose build`
5. Then tag and push both the frontend and backend to `Artifact Registry`
6. Next, create the two `Cloud Run` services for the `Streamlit` frontend and `FastAPI`backend. The frontend uses the same URL generated for the backend service to run requests.
7. Connect `Cloud Build` to the GitHub repository to set up a trigger when the `main` branch is updated.

