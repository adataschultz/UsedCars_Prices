# Use a Python-based image
#FROM nvcr.io/nvidia/rapidsai/base:24.04-cuda11.2-py3.10

FROM python:3.10-slim

WORKDIR /frontend                                                                                                     
RUN apt-get update && apt-get install -y     python3-pip                                                          
COPY /data/usedCars_trainSet.csv /data/usedCars_trainSet.csv                                                            
COPY /data/usedCars_testSet.csv /data/usedCars_testSet.csv                                                             
COPY /xgboost/model/usedcars_xgb_model.bin /xgboost/model/usedcars_xgb_model.bin                                       
COPY /xgboost/results /xgboost/results                                                                                 
COPY /catboost/model/usedcars_cat_model /catboost/model/usedcars_cat_model                                             
COPY /catboost/results /catboost/results                                                                                 
COPY /lightgbm/model/usedcars_lgbm_model.pkl /lightgbm/model/usedcars_lgbm_model.pkl                                     
COPY /lightgbm/results /lightgbm/results                                                                                
COPY /static /static                                                                                                   
COPY app.py .                                                                                                           
COPY requirements.txt requirements.txt                                                                                 
RUN pip3 install --upgrade pip --no-cache-dir -r requirements.txt

# Copy application code
COPY . .


# Expose port
EXPOSE 8501

# Run the application
CMD ["python", "run", "streamlit", "app.py"]
