# Base container
FROM apache/airflow:2.7.3

# Set working directory
WORKDIR /airflow_gcp

# Create volume and copy
VOLUME /include

# Copy data, models, code, requirements
COPY include/data/usedcars_trainset.parquet /opt/airflow/include/data/usedcars_trainset.parquet                                                        
COPY include/data/usedcars_testset.parquet /opt/airflow/include/data/usedcars_testset.parquet  
COPY include/gcp/google_credentials.json /opt/airflow/include/gcp/google_credentials.json
COPY dags/ /opt/airflow/dags/
COPY requirements.txt /requirements.txt

#USER airflow # Make sure the airflow user is set before running commands
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements.txt # --no-cache-dir good practise when installing packages using pip. It helps to keep the image lightweight

# Source: https://airflow.apache.org/docs/docker-stack/recipes.html
# Installing the GCP CLI in the container
SHELL ['/bin/bash', '-o', 'pipefail', '-e', '-u', '-x', '-c']

USER 0 
ARG CLOUD_SDK_VERSION=322.0.0
ENV GCLOUD_HOME=/home/google-cloud-sdk
export AIRFLOW_HOME=`pwd`
airflow db init
ENV AIRFLOW__CORE__XCOM_BACKEND='astro.custom_backend.astro_custom_backend.AstroCustomXcomBackend'
ENV PATH='${GCLOUD_HOME}/bin/:${PATH}'

RUN DOWNLOAD_URL='https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-${CLOUD_SDK_VERSION}-linux-x86_64.tar.gz' \
    && TMP_DIR='$(mktemp -d)' \
    && curl -fL '${DOWNLOAD_URL}' --output '${TMP_DIR}/google-cloud-sdk.tar.gz' \
    && mkdir -p '${GCLOUD_HOME}' \
    && tar xzf '${TMP_DIR}/google-cloud-sdk.tar.gz' -C '${GCLOUD_HOME}' --strip-components=1 \
    && '${GCLOUD_HOME}/install.sh' \
    --bash-completion=false \
    --path-update=false \
    --usage-reporting=false \
    --quiet \
    && rm -rf '${TMP_DIR}' \
    && gcloud --version

