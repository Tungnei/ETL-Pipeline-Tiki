from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from scripts.etl_process_to_pg import ETLProcess
import os
from scripts.config import output_dir
import pandas as pd


with DAG(
    dag_id='etl_process_toPostgres',
    start_date=datetime(2024, 11, 15),  # Thời gian bắt đầu theo giờ Việt Nam
    catchup=False,
    schedule_interval='@daily',  # Chạy hàng ngày vào lúc 8:00 theo giờ Việt Nam
) as dag:
    
    def etl_to_Pg():
        processed_file = []
        
        for filename in os.listdir(output_dir):
            if filename.startswith('data_product') and filename.endswith('.csv'):
                file_path = os.path.join(output_dir, filename)
                if filename not in processed_file:
                    ## extract and transform data
                    etl = ETLProcess()
                    try:
                        df_et = etl.transform_data(file_path)
                        print(f"Transformed data for file {filename}")
                    except Exception as e:
                        print(f"Failed to transform data for file {filename}")
                    
                    
                    ## load data to postgres
                    try:
                        etl.load_date_dim_table(df_et)
                        print(f"Loaded date_dim_table for file {filename}")
                    except Exception as e:
                        print(f"Failed to load date_dim_table for file {filename}")
                    


                    try:
                        etl.load_sales_fact_table(df_et)
                        print(f"Loaded sales_fact_table for file {filename}")
                    except Exception as e:
                        print(f"Failed to load sales_fact_table for file {filename}")
                    try:
                        etl.load_product_dim_table(df_et)
                        print(f"Loaded product_dim_table for file {filename}")
                    except Exception as e:
                        print(f"Failed to load product_dim_table for file {filename}")
                    
                    
                    print(f"Processed file {filename}")
                    
                    processed_file.append(filename)

        return processed_file








    extract_data_and_transform = PythonOperator(
        task_id='extract_data_and_transform',
        python_callable=etl_to_Pg,
    )

extract_data_and_transform