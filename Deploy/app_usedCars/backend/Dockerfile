# Use a Python-based image
#FROM nvcr.io/nvidia/rapidsai/base:24.04-cuda11.2-py3.10
FROM tiangolo/uvicorn-gunicorn:python3.10

# Specify working directory
WORKDIR /backend

RUN apt-get update && apt-get install -y \
    python3-pip

VOLUME /data
VOLUME /xgboost/model
VOLUME /catboost/model
VOLUME /lightgbm/model

COPY /data/usedCars_trainSet.csv /data/usedCars_trainSet.csv
COPY /data/usedCars_testSet.csv /data/usedCars_testSet.csv
COPY /xgboost/model/usedcars_xgb_model.bin /xgboost/model/usedcars_xgb_model.bin
COPY /catboost/model/usedcars_cat_model /catboost/model/usedcars_cat_model
COPY /lightgbm/model/usedcars_lgbm_model.pkl /lightgbm/model/usedcars_lgbm_model.pkl
COPY main.py .

ENV TRAIN_DATA_DIR=/data/usedCars_trainSet.csv
ENV TEST_DATA_DIR=/data/usedCars_testSet.csv
ENV XGB_MODEL_DIR=/xgboost/model/usedcars_xgb_model.bin
ENV CAT_MODEL_DIR=/catboost/model/usedcars_cat_model
ENV LGB_MODEL_DIR=/lightgbm/model/usedcars_lgbm_model.pkl

# Copy requirements.txt and install dependencies
COPY . .

USER root

RUN pip3 install --upgrade pip --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose port
#EXPOSE 8000
