from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from scripts.crawl_product_data import crawl
from scripts.util_minio import MinioHandler
from scripts.config import bucket_name,output_dir,output_file,file_dir
import os
with DAG(
    dag_id='extract_from_tiki_and_load_to_s3',
    start_date=datetime(2024, 11, 8),  # Thời gian bắt đầu theo giờ Việt Nam
    catchup=False,
    schedule_interval='@daily',  # Chạy hàng ngày vào lúc 8:00 theo giờ Việt Nam
) as dag:

    def crawl_data():
        output_path = os.path.join(output_dir, output_file)

        if os.path.isfile(output_path):
            print(f"File '{output_file}' đã tồn tại, không cần crawl lại.")
        else :
            crawl()
      
    def upload_to_s3():
        current_time = datetime.now().strftime("%d%m%y")
        minio_handler = MinioHandler()
        minio_handler.upload_file_to_bucket(bucket_name, file_dir,output_file)

    

    crawl_product_data = PythonOperator(
        task_id='crawl_data',
        python_callable=crawl_data,
    )
    load_to_s3 = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
    )

    crawl_product_data>>load_to_s3

  
    