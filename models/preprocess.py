import os

from sklearn.datasets import load_iris, load_wine
from pandas_datareader import wb
import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType

this_dir = os.path.dirname(os.path.abspath(__file__))
metastore_path = os.path.join(this_dir, "metastore_db")
warehouse_path = os.path.join(this_dir, "spark-warehouse")

spark = SparkSession.builder.appName('example')\
    .config('spark.sql.catalogImplementation', 'hive') \
    .config("javax.jdo.option.ConnectionURL", f"jdbc:derby:;databaseName={metastore_path};create=true") \
    .config("spark.sql.warehouse.dir", warehouse_path) \
    .enableHiveSupport().getOrCreate()

table_names = list(map(lambda x: x.name, spark.catalog.listTables()))
print('table_names:', table_names)

# Irisデータセットのテーブル(Iris)の有無を確認
if 'iris' not in table_names:
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df.to_csv('iris.csv', index=False)
    schema = StructType([
        StructField("sepal length (cm)", DoubleType(), True),
        StructField("sepal width (cm)", DoubleType(), True),
        StructField("petal length (cm)", DoubleType(), True),
        StructField("petal width (cm)", DoubleType(), True),
        StructField("target", IntegerType(), True)
    ])

    spark.read.csv('iris.csv', header=True, schema=schema).write.saveAsTable('Iris')

spark.sql("SELECT * FROM Iris LIMIT 5").show()

result = spark.sql(f"SHOW CREATE TABLE IRIS")
create_table_stmt = result.collect()[0].createtab_stmt
print('create_table_stmt:', create_table_stmt)

if 'gdppercapita' not in table_names:

    gdppc_df = wb.download(indicator='NY.GDP.PCAP.KD', country=['US', 'CN', 'JP'], start=2005, end=2022)
    gdppc_data = []
    for idx, row in gdppc_df.iterrows():
        country = idx[0]
        year = idx[1]
        gdppc = row.values[0]
        gdppc_data.append([country, year, gdppc])
    gdppc_df = pd.DataFrame(gdppc_data, columns=['Country', 'Year', 'GDPPerCapita'])
    gdppc_df.to_csv('gdppc.csv', index=False)
    schema = StructType([
        StructField("Country", StringType(), True),
        StructField("Year", IntegerType(), True),
        StructField("GDPPerCapita", DoubleType(), True),
    ])

    spark.read.csv('gdppc.csv', header=True, schema=schema).write.saveAsTable('GDPPerCapita')

spark.sql("SELECT * FROM GDPPerCapita LIMIT 5").show()

result = spark.sql(f"SHOW CREATE TABLE GDPPerCapita")
create_table_stmt = result.collect()[0].createtab_stmt
print('create_table_stmt:', create_table_stmt)
