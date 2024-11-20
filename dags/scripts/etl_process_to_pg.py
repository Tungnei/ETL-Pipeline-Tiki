from sqlalchemy import create_engine, text
import os

import pandas as pd

engine = create_engine('postgresql://myuser:mypassword@localhost:5432/mydatabase2')

class ETLProcess:
    def __init__(self):
        self.engine = engine
        self.df = None
    def transform_data(self,file_path):
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
        df['product_id'] = df['product_id'].astype('Int64')  # BIGINT
        df['product_name'] = df['product_name'].astype('str')  # TEXT
        df['brand_name'] = df['brand_name'].astype('str')  # TEXT
        df['short_description'] = df['short_description'].astype('str')  # TEXT
        df['original_price'] = df['original_price'].astype('int64')  # BIGINT
        df['price_after_voucher'] = df['price_after_voucher'].astype('int64')  # BIGINT
        df['discount_rate'] = df['discount_rate'].astype('int64')  # INTEGER
        df['discount_price'] = df['discount_price'].astype('int64')  # BIGINT
        df['quantity_sold'] = df['quantity_sold'].astype('int64')  # INTEGER
        df['rating_average'] = df['rating_average'].astype('float64')  # DOUBLE PRECISION
        df['review_count'] = df['review_count'].astype('int64')  # INTEGER
        df['warranty_info'] = df['warranty_info'].astype('str')  # TEXT
        df['return_policy'] = df['return_policy'].astype('str')  # TEXT
        df['url_img'] = df['url_img'].astype('str')  # TEXT
        df['date'] = df['date'].astype('datetime64[ns]')  # TIMESTAMP
        df['day'] = df['day'].astype('int32')  # INTEGER
        df['month'] = df['month'].astype('int32')  # INTEGER
        df['year'] = df['year'].astype('int32')  # INTEGER
        df['quarter'] = df['quarter'].astype('int32')  # INTEGER
        print("Transformed data successfully")
        self.df = df
        return self.df
    
    def load_product_dim_table(self):
        print("Loading data to destination")
        
        self.df[['product_id', 'product_name', 'brand_name', 'short_description', 'warranty_info', 'return_policy', 'url_img']].to_sql('product_dim_table', self.engine, if_exists='append', index=False)
        print("Loaded data successfully")

    def load_sales_fact_table(self):
        print("Loading data to destination")
        self.df[['product_id','date','price_after_voucher', 'original_price', 'discount_rate', 'discount_price', 'quantity_sold', 'rating_average', 'review_count']].to_sql('sales_fact_table', self.engine, if_exists='append', index=False)
        print("Loaded data successfully")
    def load_date_dim_table(self):
        print("Loading data to destination")
        date_dim_table = self.df[['date','day', 'month', 'year', 'quarter']].drop_duplicates()
        date_dim_table.to_sql('date_dim_table', self.engine, if_exists='append', index=False)
        print("Loaded data successfully")

   