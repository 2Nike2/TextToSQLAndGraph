import os

from sklearn.datasets import load_iris, load_wine
import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

this_dir = os.path.dirname(os.path.abspath(__file__))
metastore_path = os.path.join(this_dir, "metastore_db")
warehouse_path = os.path.join(this_dir, "spark-warehouse")

spark = SparkSession.builder.appName('example')\
    .config('spark.sql.catalogImplementation', 'hive') \
    .config("javax.jdo.option.ConnectionURL", f"jdbc:derby:;databaseName={metastore_path};create=true") \
    .config("spark.sql.warehouse.dir", warehouse_path) \
    .enableHiveSupport().getOrCreate()

# Irisデータセットのテーブル(Iris)の有無を確認
if 'Iris' not in spark.catalog.listTables():
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
