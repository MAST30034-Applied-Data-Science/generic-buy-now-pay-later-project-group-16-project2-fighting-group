from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
from pyspark import sql
from pyspark.sql.functions import round
from pyspark.sql import Window

# Create a spark session (which will run spark jobs)
spark = (
    SparkSession.builder.appName("MAST30034 Tutorial 1")
    .config("spark.sql.repl.eagerEval.enabled", True) 
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .getOrCreate()
)

#transaction
ori_transaction1 = spark.read.parquet('/Users/kezhang/Desktop/generic-buy-now-pay-later-project-group-16-project2-fighting-group/data/tables/transactions_20210228_20210827_snapshot').sort('order_datetime')
ori_transaction2 = spark.read.parquet('/Users/kezhang/Desktop/generic-buy-now-pay-later-project-group-16-project2-fighting-group/data/tables/transactions_20210828_20220227_snapshot').sort('order_datetime')
ori_transaction3 = spark.read.parquet('/Users/kezhang/Desktop/generic-buy-now-pay-later-project-group-16-project2-fighting-group/data/tables/transactions_20220228_20220828_snapshot').sort('order_datetime')
#consumer
tbl_consumer = spark.read.option("delimiter", "|").option("header",True).csv("/Users/kezhang/Desktop/generic-buy-now-pay-later-project-group-16-project2-fighting-group/data/tables/tbl_consumer.csv")
consumer_detail = spark.read.parquet('/Users/kezhang/Desktop/generic-buy-now-pay-later-project-group-16-project2-fighting-group/data/tables/consumer_user_details.parquet')
#fraud
consumer_fraud = spark.read.option("header",True).csv('/Users/kezhang/Desktop/generic-buy-now-pay-later-project-group-16-project2-fighting-group/data/tables/consumer_fraud_probability.csv')

#transaction
ori_transaction = ori_transaction1.union(ori_transaction2)
ori_transaction = ori_transaction.union(ori_transaction3)
ori_transaction = ori_transaction.drop('order_id')
ori_transaction.limit(3)

#delete outlier
wind = Window.partitionBy('merchant_abn')

q1= F.expr('percentile_approx(dollar_value, 0.25)')
q3= F.expr('percentile_approx(dollar_value, 0.75)')

testq = ori_transaction.withColumn('q1', q1.over(wind))
testq = testq.withColumn('q3', q3.over(wind))

testq = testq.withColumn('IQR', testq['q3']-testq['q1'])

dele_out = testq.where((testq["dollar_value"] <= testq["q1"]+1.5*testq["IQR"]) & (testq["dollar_value"] >= testq["q1"]-1.5*testq["IQR"]))
ori_transaction = dele_out.drop('q1','q3','IQR','order_id')

ori_transaction = ori_transaction.withColumn(
    "whether_bigorder",
    F.when(F.col('dollar_value')>=10000, 1).otherwise(0))

#consumer
consumer = consumer_detail.join(tbl_consumer, consumer_detail.consumer_id == tbl_consumer.consumer_id).drop(tbl_consumer.consumer_id)
consumer = consumer.select('user_id', 'postcode')

# fraud
consumer_fraud = consumer_fraud.select(col("user_id"),col("fraud_probability"),to_date(col("order_datetime"),"yyyy-MM-dd").alias("date"))

consumer_fraud_1 = consumer_fraud.groupBy('user_id').agg(F.avg('fraud_probability').alias('average_prob_con'))
consumer_fraud_final = consumer_fraud_1.withColumn(
    "whether_fraud",
    F.when(F.col('average_prob_con')>=70, 1).otherwise(0))

consumer = consumer.join(consumer_fraud_final, consumer.user_id == consumer_fraud_final.user_id).drop(consumer_fraud_final.user_id)

print(consumer)

transaction = ori_transaction.join(consumer, ori_transaction.user_id == consumer.user_id).drop(consumer.user_id)
transaction = transaction.sort(transaction.user_id)

print(transaction)

###############

merchant_fraud = spark.read.option("header",True).csv('/Users/kezhang/Desktop/generic-buy-now-pay-later-project-group-16-project2-fighting-group/data/tables/merchant_fraud_probability.csv')
tbl_merchants = spark.read.parquet('/Users/kezhang/Desktop/generic-buy-now-pay-later-project-group-16-project2-fighting-group/data/tables/tbl_merchants.parquet')

merchant_fraud = merchant_fraud.select(col("merchant_abn"),col("fraud_probability"),to_date(col("order_datetime"),"yyyy-MM-dd").alias("date"))

merchant_fraud_group = merchant_fraud.groupBy('merchant_abn').agg(F.count('fraud_probability').alias('fraud_count_abn'))

tbl_merchants_pd = tbl_merchants.toPandas()
for i in range(int(tbl_merchants_pd['tags'].count())):
    tbl_merchants_pd['tags'].iloc[i] = tbl_merchants_pd['tags'].iloc[i].replace(r'[', r'(').replace(r']', r')')

merchant_tags = tbl_merchants_pd['tags'].str.split(')', expand=True)

for row in range(int(len(merchant_tags))):
    for col in range(3):
        merchant_tags.iloc[row,col] = merchant_tags.iloc[row,col].replace(r'((', r'').replace(r', (', r'').replace(r'take rate:', r'')
merchant_tags.rename(columns = {0 : 'Store type', 1 : 'Revenue levels', 2 : 'Take rate'}, inplace = True)
merchant_tags = merchant_tags[['Store type', 'Revenue levels', 'Take rate']]

tbl_merchants_pd[['Store type', 'Revenue levels', 'Take rate']] = merchant_tags[['Store type', 'Revenue levels', 'Take rate']]
#merchant.drop(columns=['tags'])
tbl_merchants_pd['Store type'] = tbl_merchants_pd['Store type'].str.lower()

for i in range(len(tbl_merchants_pd)):
    tbl_merchants_pd['Store type'][i] = ' '.join(tbl_merchants_pd['Store type'][i].split())


merchants = tbl_merchants_pd.drop(columns = ['name','tags'])
spark_session = sql.SparkSession.builder.appName("pdf to sdf").getOrCreate()
merchants = spark_session.createDataFrame(merchants)
print(merchants)

###################################

grouped_transaction = transaction.groupBy('merchant_abn').agg(
    F.sum('dollar_value').alias('Amount'), F.count('dollar_value').alias('Count'), 
    F.sum('whether_bigorder').alias('count_of_bigorder')).sort('merchant_abn')
grouped_transaction.drop(F.col('order_id'))
grouped_transaction = grouped_transaction.withColumn('Avg_amount_monthly', round(grouped_transaction['Amount']/12, 2))
grouped_transaction = grouped_transaction.withColumn('Avg_count_monthly', round(grouped_transaction['Count']/12, 2))
grouped_transaction = grouped_transaction.withColumn('Order_avg_value', round(grouped_transaction.Amount/grouped_transaction.Count,2))
grouped_transaction = grouped_transaction.drop('Amount','Count')

merchant_data1 = merchants.join(grouped_transaction, merchants.merchant_abn == grouped_transaction.merchant_abn).drop(grouped_transaction.merchant_abn)
merchant_data1 = merchant_data1.drop('Amount','Count')


ori_transaction_1 = transaction.groupby('merchant_abn','user_id').agg(
    F.count('user_id').alias('count'), 
    F.avg('average_prob_con').alias('avg_prob_fraud_cus'),
    F.avg('whether_fraud').alias('whether_fraud'))

o_t = ori_transaction_1.groupby('merchant_abn').agg(
    F.count('user_id').alias('cnt'), 
    F.avg('avg_prob_fraud_cus').alias('avg_prob_fraud_cus'),
    F.sum('whether_fraud').alias('num_of_fraud'))
    
cus_per_mon = o_t.withColumn('count_cus_per_mon', round(o_t['cnt']/12, 2))
cus_per_mon = cus_per_mon.drop('cnt')

ori_transaction_2 = transaction.groupby('merchant_abn', 'user_id').count()
o_t_2 = transaction.groupby('merchant_abn').agg(F.count('user_id').alias('total_c_num'))
ori_con = ori_transaction_2.join(o_t_2, ori_transaction_2.merchant_abn == o_t_2.merchant_abn).drop(o_t_2.merchant_abn)

ori_con_drop = ori_con.withColumn(
    "fixed_cus_num",
    F.when(F.col("count") > 5, 1).otherwise(0))

ori_con_fix = ori_con_drop.groupby('merchant_abn').agg(F.sum('fixed_cus_num').alias('fix_cus_num'), F.avg('total_c_num').alias('total_cus_num'))


ori_con_fix_prob = ori_con_fix.withColumn("fix_cus_prob", ori_con_fix.fix_cus_num/ori_con_fix.total_cus_num)

user_info = cus_per_mon.join(ori_con_fix_prob, cus_per_mon.merchant_abn == ori_con_fix_prob.merchant_abn).drop(ori_con_fix_prob.merchant_abn)
user_info = user_info.drop('total_cus_num','fix_cus_num')

user_group_transcation = grouped_transaction.join(user_info, grouped_transaction.merchant_abn == user_info.merchant_abn).drop(user_info.merchant_abn)

merchant_abn_and_consumer_id = transaction['merchant_abn', 'user_id']
user_id_and_postcode = consumer[['postcode','user_id']]

merchant_and_consumer_postcode = merchant_abn_and_consumer_id.join(user_id_and_postcode,['user_id'])
merchant_and_consumer_postcode = merchant_and_consumer_postcode['merchant_abn', 'postcode']

#https://stackoverflow.com/questions/36654162/mode-of-grouped-data-in-pyspark
counts = merchant_and_consumer_postcode.groupBy(['merchant_abn', 'postcode']).count().alias('counts')
merchant_postcode = (counts
          .groupBy('merchant_abn')
          .agg(F.max(F.struct(F.col('count'),
                              F.col('postcode'))).alias('max'))
          .select(F.col('merchant_abn'), F.col('max.postcode'))
         )

final_merchant_info = user_group_transcation.join(merchant_postcode, user_group_transcation.merchant_abn == merchant_postcode.merchant_abn).drop(merchant_postcode.merchant_abn)
print(final_merchant_info)














