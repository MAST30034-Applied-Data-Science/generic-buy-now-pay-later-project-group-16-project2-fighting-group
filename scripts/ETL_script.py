from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
import numpy as np
from pyspark.sql.functions import *
import pandas as pd
import os
import zipfile
import geopandas as gpd
from dbfread import DBF
from urllib.request import urlretrieve
from urllib.error import HTTPError
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from operator import add
from functools import reduce
from pyspark.sql.functions import round
from datetime import datetime


# Create a spark session (which will run spark jobs)
spark = (
    SparkSession.builder.appName("MAST30034 Tutorial 1")
    .config("spark.sql.repl.eagerEval.enabled", True) 
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .getOrCreate()
)

# ---------------------------------------
## OBTAINED THE PROCESSED TRANSACTION data
# input and read the given transaction datasets

while True:
    try: 
        num_folders = int(input('Please enter the number of transaction folders in total: '))
        if num_folders <= 0:
            print("Input must be a positive integer, try again")
            continue
        break
    except ValueError:
        print('Input must be a positive integer, try again')
        continue

while True:
    try: 
        transaction_name = input('Please enter the name of the first folder: ')
        ori_transaction = spark.read.parquet('data/tables/'+str(transaction_name)).sort('order_datetime')
        for i in range(int(num_folders)-1):
            transaction_name = input('Please enter the name of the next folder: ')
            ori_transaction_rest = spark.read.parquet('data/tables/'+str(transaction_name)).sort('order_datetime')
            ori_transaction = ori_transaction.union(ori_transaction_rest)
    except:
        print('Please enter the correct folder name, start from the first folder, try again')
        continue
    else:
        break

while True:
    try: 
        start_date_str = str(input('Please enter the start date of the transaction (in yyyy-mm-dd format): '))
        end_date_str = str(input('Please enter the end date of the transaction (in yyyy-mm-dd format): '))
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except:
        print("Correct date format should be yyyy-mm-dd, try again")
        continue
    else:
        break

number_of_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
with open('data/number_of_months.txt', 'w') as f:
    f.write(str(number_of_months))

ori_transaction = ori_transaction.drop('order_id')
ori_transaction = ori_transaction.dropna(how='any')
ori_transaction = ori_transaction.where(
        # clean the data outside the date range
        (F.col('order_datetime') >= start_date_str)& 
        (F.col('order_datetime') <= end_date_str))

# read the given consumer datasets
tbl_consumer = spark.read.option("delimiter", "|").option("header",True).csv("data/tables/tbl_consumer.csv")
consumer_detail = spark.read.parquet('data/tables/consumer_user_details.parquet')

# read the fraud dataset
consumer_fraud = spark.read.option("header",True).csv('data/tables/consumer_fraud_probability.csv')

# delete outliers
wind = Window.partitionBy('merchant_abn')
q1= F.expr('percentile_approx(dollar_value, 0.25)')
q3= F.expr('percentile_approx(dollar_value, 0.75)')
testq = ori_transaction.withColumn('q1', q1.over(wind))
testq = testq.withColumn('q3', q3.over(wind))
testq = testq.withColumn('IQR', testq['q3']-testq['q1'])
dele_out = testq.where((testq["dollar_value"] <= testq["q1"]+1.5*testq["IQR"]) & (testq["dollar_value"] >= testq["q1"]-1.5*testq["IQR"]))
ori_transaction = dele_out.drop('q1','q3','IQR','order_id')

# give a definition, if a single transaction is over $10000, this is a big order
big_order_value = 10000
ori_transaction = ori_transaction.withColumn(
    "whether_bigorder",
    F.when(F.col('dollar_value')>=big_order_value, 1).otherwise(0))

# consumer datasets
consumer = consumer_detail.join(tbl_consumer, consumer_detail.consumer_id == tbl_consumer.consumer_id).drop(tbl_consumer.consumer_id)
consumer = consumer.select('user_id', 'postcode')

# change to the date format
consumer_fraud = consumer_fraud.select(col("user_id"),col("fraud_probability"),to_date(col("order_datetime"),"yyyy-MM-dd").alias("date"))

# filter the range for the fraud data
consumer_fraud = consumer_fraud.where(
        # clean the data outside the date range
        (F.col('date') >= start_date_str)& 
        (F.col('date') <= end_date_str))

consumer_fraud_grouped = consumer_fraud.groupBy('user_id').agg(F.avg('fraud_probability').alias('average_prob_con'))
consumer_fraud_final = consumer_fraud_grouped.withColumn(
    "whether_fraud",
    F.when(F.col('average_prob_con')>=70, 1).otherwise(0))

consumer_final = consumer.join(consumer_fraud_final, consumer.user_id == consumer_fraud_final.user_id).drop(consumer_fraud_final.user_id).fillna(0).fillna(0)
consumer_final.write.mode('overwrite').parquet(f"data/curated/final_consumer.parquet")

transaction = ori_transaction.join(consumer_final, ori_transaction.user_id == consumer_final.user_id).drop(consumer_final.user_id)
transaction = transaction.sort(transaction.user_id)
transaction.write.mode('overwrite').parquet(f"data/curated/final_transaction.parquet")

# ------------------------------------
## OBTAIN THE PROCESSED MERCHANT data
# read the given merchant datasets
merchant_fraud = spark.read.option("header",True).csv('data/tables/merchant_fraud_probability.csv')
tbl_merchants = spark.read.parquet('data/tables/tbl_merchants.parquet')

# check the range of date
merchant_fraud = merchant_fraud.select(col("merchant_abn"),col("fraud_probability"),to_date(col("order_datetime"),"yyyy-MM-dd").alias("date"))

# filter the range for the fraud data
merchant_fraud = merchant_fraud.where(
        # clean the data outside the date range
        (F.col('date') >= start_date_str)& 
        (F.col('date') <= end_date_str))
merchant_fraud_grouped = merchant_fraud.groupBy('merchant_abn').agg(F.count('fraud_probability').alias('fraud_count_abn'))

# preprocess the given merchant datasets
tbl_merchants_pd = tbl_merchants.toPandas()
for i in range(int(tbl_merchants_pd['tags'].count())):
    tbl_merchants_pd['tags'].iloc[i] = tbl_merchants_pd['tags'].iloc[i].replace(r'[', r'(').replace(r']', r')')
tbl_merchants_pd['tags'] = tbl_merchants_pd['tags'].str.lower()

# split the column into three columns and give names to the columns
merchant_tags = tbl_merchants_pd['tags'].str.split(')', expand=True)
for row in range(int(len(merchant_tags))):
    for col in range(3):
        merchant_tags.iloc[row,col] = merchant_tags.iloc[row,col].replace(r'((', r'').replace(r', (', r'').replace(r'take rate:', r'')
merchant_tags.rename(columns = {0 : 'Store_type', 1 : 'Revenue_levels', 2 : 'Take_rate'}, inplace = True)
merchant_tags = merchant_tags[['Store_type', 'Revenue_levels', 'Take_rate']]

tbl_merchants_pd[['Store_type', 'Revenue_levels', 'Take_rate']] = merchant_tags[['Store_type', 'Revenue_levels', 'Take_rate']]

for i in range(len(tbl_merchants_pd)):
    tbl_merchants_pd['Store_type'][i] = ' '.join(tbl_merchants_pd['Store_type'][i].split())

# export the merchant dataset
merchants = tbl_merchants_pd[['merchant_abn', 'Store_type', 'Revenue_levels', 'Take_rate']]
merchants.to_parquet(f"data/curated/final_merchant.parquet")