
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
# Create a spark session (which will run spark jobs)
spark = (
    SparkSession.builder.appName("MAST30034 Tutorial 1")
    .config("spark.sql.repl.eagerEval.enabled", True) 
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .getOrCreate()
)

# ------------------------------------
## OBTAINED THE PROCESSED TRANSACTION DATA
# read the given transaction datasets
ori_transaction1 = spark.read.parquet('data/tables/transactions_20210228_20210827_snapshot').sort('order_datetime')
ori_transaction2 = spark.read.parquet('data/tables/transactions_20210828_20220227_snapshot').sort('order_datetime')
ori_transaction3 = spark.read.parquet('data/tables/transactions_20220228_20220828_snapshot').sort('order_datetime')

# read the given consumer datasets
tbl_consumer = spark.read.option("delimiter", "|").option("header",True).csv("data/tables/tbl_consumer.csv")
consumer_detail = spark.read.parquet('data/tables/consumer_user_details.parquet')

# read the fraud dataset
consumer_fraud = spark.read.option("header",True).csv('data/tables/consumer_fraud_probability.csv')

# merge all the transaction data
ori_transaction = ori_transaction1.union(ori_transaction2)
ori_transaction = ori_transaction.union(ori_transaction3)
ori_transaction = ori_transaction.drop('order_id')
ori_transaction = ori_transaction.dropna(how='any')
ori_transaction = ori_transaction.filter("order_datetime >= '2021-02-28'")
ori_transaction = ori_transaction.filter("order_datetime <= '2022-08-28'")


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
        (F.col('date') >= '2021-02-28')& 
        (F.col('date') <= '2022-08-28'))

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
## OBTAIN THE PROCESSED MERCHANT DATA
# read the given merchant datasets
merchant_fraud = spark.read.option("header",True).csv('data/tables/merchant_fraud_probability.csv')
tbl_merchants = spark.read.parquet('data/tables/tbl_merchants.parquet')

# check the range of date
merchant_fraud = merchant_fraud.select(col("merchant_abn"),col("fraud_probability"),to_date(col("order_datetime"),"yyyy-MM-dd").alias("date"))

# filter the range for the fraud data
merchant_fraud = merchant_fraud.where(
        # clean the data outside the date range
        (F.col('date') >= '2021-02-28')& 
        (F.col('date') <= '2022-08-28'))
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

# ------------------------------------
## RETRIEVE EXTERNAL DATA
output_relative_dir = 'data/'

if not os.path.exists(output_relative_dir):
    os.makedirs(output_relative_dir)
    
for target_dir in ('external','outer'):
    if not os.path.exists(output_relative_dir + target_dir):
        os.makedirs(output_relative_dir + target_dir)

# Download census data
url = "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_AUS_short-header.zip"#year-month.parquet
output_dir = "data/outer/census.zip"
urlretrieve(url, output_dir) 
files = zipfile.ZipFile('data/outer/census.zip','r')

for file in files.namelist():
    files.extract(file, f"data/outer/census")

types = ["A", "B"]
for type in types:
    G04= spark.read.option("header",True).csv(f'data/outer/census/2021 Census GCP Statistical Area 2 for AUS/2021Census_G04{type}_AUST_SA2.csv') #read the parquet 
    G04 = G04.na.drop(subset=G04.columns)
    G04.write.mode("overwrite").parquet(f"data/external/census_data_2021_04{type}.parquet")

types = ["A", "B", "C"]
for type in types:
    G17= spark.read.option("header",True).csv(f'data/outer/census/2021 Census GCP Statistical Area 2 for AUS/2021Census_G17{type}_AUST_SA2.csv') #read the parquet 
    G17 = G17.na.drop(subset=G17.columns)

    G17.write.mode("overwrite").parquet(f"data/external/census_data_2021_17{type}.parquet")

# Download postcode and SA2 data
url = "http://github.com/matthewproctor/australianpostcodes/zipball/master"
output_dir = "data/outer/total_pto2.zip"
urlretrieve(url, output_dir) 

files = zipfile.ZipFile('data/outer/total_pto2.zip','r')
for file in files.namelist():
    files.extract(file, f"data/outer/total_pto2")

post_sa2= spark.read.option("header",True).csv('data/outer/total_pto2/matthewproctor-australianpostcodes-6f8a994/australian_postcodes.csv') 
post_sa2 = post_sa2.select("id", "postcode", "Lat_precise", "Long_precise", "SA2_MAINCODE_2016")
post_sa2 = post_sa2.na.drop()

# get the average latitude and longitude of the postcode if it have duplicates
post_sa2 = post_sa2\
    .withColumn("Lat_double", post_sa2["Lat_precise"].cast("double"))\
    .withColumn("Long_double", post_sa2["Long_precise"].cast("double"))
post_sa2 = post_sa2\
    .groupBy("postcode", "SA2_MAINCODE_2016")\
    .agg(F.avg("Lat_double").alias("avg_lat"), F.avg("Long_double").alias("avg_long"))

# Download SA2 2021 information
url = "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip" 
output_dir = "data/outer/2021sa2_shapefile.zip"

urlretrieve(url, output_dir) 
print("complete")

files = zipfile.ZipFile('data/outer/2021sa2_shapefile.zip','r')
for file in files.namelist():
    files.extract(file, f"data/outer/2021sa2_shapefile")

# Download shapefile for each SA2 region
def get_shapefile(url):
    geojson_option = "?_profile=oai&_mediatype=application/geo+json"
    try:
        shape = str(gpd.read_file(url + geojson_option).iat[0,-1])
    except HTTPError:
        shape = ""
    return shape

get_shapefile_udf = udf(lambda a: get_shapefile(a),StringType())

path = r'data/outer/2021sa2_shapefile/SA2_2021_AUST_GDA2020.dbf' 
table = DBF(path)
sa2_pd_temp = pd.DataFrame(iter(table))
sa2_2021_temp = spark.createDataFrame(sa2_pd_temp) 

# extract only the data from Victoria state for visualisation 
sa2_2021_vic = sa2_2021_temp.filter(F.col("STE_NAME21")=="Victoria")
sa2_2021_vic_w_geo = sa2_2021_vic.withColumn("geometry", get_shapefile_udf(F.col("LOCI_URI21")))
sa2_2021_vic_w_geo = sa2_2021_vic_w_geo.select("SA2_CODE21", "SA2_NAME21", "geometry")

# Download file with 2016 SA2 info and 2021 SA2 info
url = "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/correspondences/CG_SA2_2016_SA2_2021.csv"
output_dir = "data/outer/correspondences.csv"
urlretrieve(url, output_dir) 

correspondences = spark.read.option("header",True).csv('data/outer/correspondences.csv') #read the parquet 
correspondences = correspondences.na.drop()

# Find 2016 post code and 2021 sa2 correspondence
post_sa2_2021 = correspondences.join(post_sa2,correspondences.SA2_MAINCODE_2016 == post_sa2.SA2_MAINCODE_2016,"left") 

# JOIN tables: this will be the final SA2 and postcode correnpondence file
post_sa2_2021 = post_sa2_2021.drop("SA2_NAME_2016", "SA2_MAINCODE_2016")
post_sa2_2021.write.mode('overwrite').parquet("data/external/postcode_sa2_conrrespondences.parquet")


# ------------------------------------
## PROCESS CENSUS AGE AND INCOME DATA
# read the external census datasets
census_04A_age = spark.read.parquet('data/external/census_data_2021_04A.parquet')
census_04B_age = spark.read.parquet('data/external/census_data_2021_04B.parquet')
census_17A_income = spark.read.parquet('data/external/census_data_2021_17A.parquet')
census_17B_income = spark.read.parquet('data/external/census_data_2021_17B.parquet')
census_17C_income = spark.read.parquet('data/external/census_data_2021_17C.parquet')
postcode = post_sa2

# Merge the age dataframe
Age_Data = census_04A_age.join(census_04B_age, census_04A_age.SA2_CODE_2021 == census_04B_age.SA2_CODE_2021).drop(census_04B_age.SA2_CODE_2021)

# Select the column in the age dataframe (18-65) and calculate the sum
Age_Data_select = Age_Data.select(Age_Data.columns[0::18])
Age_Data_total = Age_Data_select.withColumn('result',reduce(add, [F.col(x) for x in Age_Data_select.columns[5:14]]))
Age_Data_total = Age_Data_total.select("SA2_CODE_2021","result","Tot_P")

# Calculate the percentage of the targeted age group of all people
Age_Data_rate = Age_Data_total.withColumn("age_percentage", Age_Data_total.result /Age_Data_total.Tot_P)
Age_Data_rate = Age_Data_rate.select("SA2_CODE_2021","age_percentage")

# Combine income table
income = census_17A_income.join(census_17B_income, census_17A_income.SA2_CODE_2021 == census_17B_income.SA2_CODE_2021).drop(census_17B_income.SA2_CODE_2021)
income_data = income.join(census_17C_income, income.SA2_CODE_2021 == census_17C_income.SA2_CODE_2021).drop(census_17C_income.SA2_CODE_2021)

# Select the column in the income that above Australia medium income and calculate the sum
income_data_del = income_data.select(income_data.columns[0::10])
income_data_select = income_data_del.withColumn('result',reduce(add, [F.col(x) for x in income_data_del.columns[44:50]]))
income_Data_total = income_data_select.select("SA2_CODE_2021","result","P_Tot_Tot")

# Calculate the percentage of the targeted income group of all people
income_Data_total = income_Data_total.withColumn("income_percentage", income_Data_total.result / income_Data_total.P_Tot_Tot)
income_Data_total = income_Data_total.select("SA2_CODE_2021","income_percentage")

# Combine income data and the age data to get final census data
census_data = income_Data_total.join(Age_Data_rate, income_Data_total.SA2_CODE_2021 == Age_Data_rate.SA2_CODE_2021).drop(Age_Data_rate.SA2_CODE_2021)
census_data.write.mode('overwrite').parquet('data/curated/final_census.parquet')

