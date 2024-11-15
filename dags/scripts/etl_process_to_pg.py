from sqlalchemy import create_engine, text
import os
from config import output_dir      ### output file path when to crawl data

import pandas as pd

engine = create_engine('postgresql://myuser:mypassword@localhost:5432/mydatabase')

class ETLProcess:
    def __init__(self):
        self.engine = engine

    def transform_data(file_path):
        print("Transforming data")
        ## transform data to fit with date_dim_table
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        
        ## transform data to fit with product_dim_table and sales_fact_table
        df = df.rename(columns={
        'id': 'product_id',
        'product_name': 'product_name',
        'brand_name': 'brand_name',
        'short_description': 'short_description',
        'original_price': 'original_price',
        'price_after_voucher': 'price_after_voucher',
        'discount_rate': 'discount_rate',
        'discount_price': 'discount_price',
        'quantity_sold': 'quantity_sold',
        'rating_average': 'rating_average',
        'review_count': 'review_count',
        'warranty_info': 'warranty_info',
        'return_policy': 'return_policy',
        'url_img': 'url_img'
    })
        print("Transformed data successfully")
        return df
    
    def load_product_dim_table(self,df):
        print("Loading data to destination")
        df[['product_id', 'product_name', 'brand_name', 'short_description', 'warranty_info', 'return_policy', 'url_img']].to_sql('product_dim_table', self.engine, if_exists='append', index=False)
        print("Loaded data successfully")

    def load_sales_fact_table(self,df):
        print("Loading data to destination")
        df[['product_id', 'year', 'month', 'day', 'price_after_voucher', 'original_price', 'discount_rate', 'discount_price', 'quantity_sold', 'rating_average', 'review_count']].to_sql('sales_fact_table', self.engine, if_exists='append', index=False)
        print("Loaded data successfully")
    def load_date_dim_table(self,df):
        print("Loading data to destination")
        df[['date', 'day', 'month', 'year', 'quarter']].to_sql('date_dim_table', self.engine, if_exists='append', index=False)
        print("Loaded data successfully")

   