'''
Airflow DAG for uploading data from  local file system to Google Cloud Storage and BigQuery
'''

from __future__ import annotations
import os
from datetime import datetime
import airflow
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator, GCSDeleteBucketOperator
#from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateEmptyDatasetOperator,
    BigQueryCreateEmptyTableOperator,
    BigQueryDeleteDatasetOperator,
)
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT_ID = 'deploy-models-gcp'
BUCKET_NAME = 'usedscars-gcp-pipeline'
DAG_ID = 'gcp_upload_gcs_bigquery'

PATH_SAVED_TRAIN = '/opt/airflow/include/data/usedcars-trainset.parquet'
PATH_SAVED_TEST = '/opt/airflow/include/data/usedcars-testset.parquet'

PARQUET_TRAIN = 'usedcars-trainset.parquet'
PARQUET_TEST = 'usedcars-testset.parquet'

TRAINSET_NAME = os.environ.get("GCP_DATASET_NAME", 'usedcars_train')
TRAINTABLE_NAME = os.environ.get("GCP_TABLE_NAME", 'train')

TESTSET_NAME = os.environ.get("GCP_DATASET_NAME", 'usedcars_test')
TESTTABLE_NAME = os.environ.get("GCP_TABLE_NAME", 'test')

with DAG(
    DAG_ID,
    schedule='@daily',
    start_date=datetime(2024, 7, 3),
    catchup=False,
    tags=['gcs', 'upload_data'],
) as dag:
    # [START howto_operator_gcs_create_bucket]
    create_bucket = GCSCreateBucketOperator(
        task_id='create_bucket',
        bucket_name=BUCKET_NAME,
        project_id=PROJECT_ID,
    )    
    gcs_upload_train = LocalFilesystemToGCSOperator(
        task_id='upload_train_file',
        src=PATH_SAVED_TRAIN,
        dst=PARQUET_TRAIN,
        bucket=BUCKET_NAME,
    )
    # [START howto_operator_local_filesystem_to_gcs]
    gcs_upload_test = LocalFilesystemToGCSOperator(
        task_id='upload_test_file',
        src=PATH_SAVED_TEST,
        dst=PARQUET_TEST,
        bucket=BUCKET_NAME,
    )    
    # [END howto_operator_local_filesystem_to_gcs]

#     # [START howto_operator_gcs_download_file_task]
#     download_file = GCSToLocalFilesystemOperator(
#         task_id='download_file',
#         object_name=FILE_NAME,
#         bucket=BUCKET_NAME,
#         filename=PATH_TO_SAVED_FILE,
#     )
#     # [END howto_operator_gcs_download_file_task]

     # [START howto_operator_gcs_delete_bucket]
    #delete_bucket = GCSDeleteBucketOperator(task_id='delete_bucket', bucket_name=BUCKET_NAME)
     # [END howto_operator_gcs_delete_bucket]
    #delete_bucket.trigger_rule = TriggerRule.ALL_DONE

    create_train_dataset = BigQueryCreateEmptyDatasetOperator(
        task_id='create_train_set', dataset_id=TRAINSET_NAME, project_id=PROJECT_ID
    )
    create_test_dataset = BigQueryCreateEmptyDatasetOperator(
        task_id='create_test_set', dataset_id=TESTSET_NAME, project_id=PROJECT_ID
    )

    # [START howto_operator_gcs_to_bigquery]
    load_parquet_train = GCSToBigQueryOperator(
        task_id='gcs_to_bigquery_train',
        bucket=BUCKET_NAME,
        source_objects=['usedcars-trainset.parquet'],
        source_format="PARQUET",
        destination_project_dataset_table=f'{TRAINSET_NAME}.{TRAINTABLE_NAME}',
        schema_fields=[
            {'name': 'price', 'type': 'FLOAT64', 'mode': 'NULLABLE'}, 
            {'name': 'back_legroom', 'type': 'FLOAT64',  'mode': 'NULLABLE'},
            {'name': 'body_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'city_fuel_economy', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'daysonmarket', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'engine_displacement', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'front_legroom', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'fuel_tank_volume', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'fuel_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'height', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'highway_fuel_economy', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'is_new', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'length', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'listing_color', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'maximum_seating', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'mileage', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'savings_amount', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'transmission', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'wheel_system_display', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'wheelbase', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'width', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'year', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'State', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'torque', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'torque_rpm', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'horsepower', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'horsepower_rpm', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'listed_date_yearMonth', 'type': 'STRING', 'mode': 'NULLABLE'},
        ],
        autodetect='True',
        write_disposition='WRITE_TRUNCATE',
    )
    # [END howto_operator_gcs_to_bigquery]

    # [START howto_operator_gcs_to_bigquery]
    load_parquet_test = GCSToBigQueryOperator(
        task_id='gcs_to_bigquery_test',
        bucket=BUCKET_NAME,
        source_objects=['usedcars-testset.parquet'],
        source_format="PARQUET",
        destination_project_dataset_table=f'{TESTSET_NAME}.{TESTTABLE_NAME}',
        schema_fields=[
            {'name': 'price', 'type': 'FLOAT64', 'mode': 'NULLABLE'}, 
            {'name': 'back_legroom', 'type': 'FLOAT64',  'mode': 'NULLABLE'},
            {'name': 'body_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'city_fuel_economy', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'daysonmarket', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'engine_displacement', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'front_legroom', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'fuel_tank_volume', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'fuel_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'height', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'highway_fuel_economy', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'is_new', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'length', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'listing_color', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'maximum_seating', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'mileage', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'savings_amount', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'transmission', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'wheel_system_display', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'wheelbase', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'width', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'year', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'State', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'torque', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'torque_rpm', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'horsepower', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'horsepower_rpm', 'type': 'FLOAT64', 'mode': 'NULLABLE'},
            {'name': 'listed_date_yearMonth', 'type': 'STRING', 'mode': 'NULLABLE'},
        ],
        autodetect='True',
        write_disposition='WRITE_TRUNCATE',
    )      
    # [END howto_operator_gcs_to_bigquery]

    (
        # Airflow pipeline steps
        create_bucket
        >> gcs_upload_train
        >> gcs_upload_test
        #>> download_file
        >> create_train_dataset 
        >> create_test_dataset
        >> load_parquet_train
        >> load_parquet_test
        #>> delete_bucket
        #>> delete_train_dataset
        #>> delete_test_dataset
    )   

    # This test needs watcher in order to properly mark success/failure
    # when 'tearDown' task with trigger rule is part of the DAG
    #list(dag.tasks) >> watcher()


# Needed to run the example DAG with pytest (see: tests/system/README.md#run_via_pytest)
#test_run = get_test_run(dag)
