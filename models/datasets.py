import os

from sklearn.datasets import load_iris, load_wine
import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from pyspark.sql import SparkSession

this_dir = os.path.dirname(os.path.abspath(__file__))
metastore_path = os.path.join(this_dir, "metastore_db")
warehouse_path = os.path.join(this_dir, "spark-warehouse")

spark = SparkSession.builder.appName('dataset')\
    .config('spark.sql.catalogImplementation', 'hive') \
    .config("javax.jdo.option.ConnectionURL", f"jdbc:derby:;databaseName={metastore_path};create=true") \
    .config("spark.sql.warehouse.dir", warehouse_path) \
    .enableHiveSupport().getOrCreate()

select_table_prompt = """\
与えられた指示に対して最も関連のあるテーブルを下記から選択して下さい。

制約:
・出力をそのまま他処理の引数に使う為、余計な文言を入れずにテーブル名のみを出力して下さい。
・必ずどれかのテーブルを選択して下さい。

指示: {instruction}

テーブル一覧:
Iris, GDPPerCapita

"""
select_table_template = PromptTemplate.from_template(select_table_prompt)


# 円グラフでは、ラベルの列のデータ型が文字列でないといけない上に、
# PySparkがPandasのデータフレームに変換する際に勝手にデータ型を変えてしまう恐れがあるので、
# ラベルの列名を指定して、この列名を使ってデータ型を文字列で明示的に変換する処理を実装して対応。
create_sql_prompt = """\
与えられた指示に適切なSQLクエリをテーブル定義を利用しつつ回答して下さい。

制約:
・出力をそのままSQL文として実行するので、余計な文言を入れずにSQL文のみを出力して下さい。
・インプリシットカラムアンチパターンは止めて出力の列名は必ず明記して下さい。
・Google Chartsの描画でラベルの列が一番左に来ないといけない場合があります。
　その場合、Labelや凡例の列は必ず列名を「Label」或いは「Label」を含む文字列の列名にして下さい。
　ただし、ラベル列が必要でないグラフもあるので考慮して下さい。
・指示に対して適切と思われるソートも行って下さい。

出力例:
SELECT COLUMN1 AS Label, COLUMN2 FROM TABLEX WHERE COLUMN3 = "XXX";

指示: {instruction}

テーブル定義: 
{table_definition}

"""
create_sql_template = PromptTemplate.from_template(create_sql_prompt)

llm = ChatOpenAI(model='gpt-4')

create_sql_chain = create_sql_template | llm | StrOutputParser()

select_table_chain = select_table_template | llm | StrOutputParser()

# テーブル定義の取得
def get_table_definition(table):

    result = spark.sql(f"SHOW CREATE TABLE {table}")
    create_table_stmt = result.collect()[0].createtab_stmt
    print('create_table_stmt:', create_table_stmt)
    return create_table_stmt

# データの取得
def get_sql_data(instruction):

    # テーブルの選択
    target_table = select_table_chain.invoke({'instruction': instruction})
    print('target_table:', target_table)

    # テーブル定義の取得
    table_definition = get_table_definition(target_table)

    # SQL生成
    create_sql = create_sql_chain.invoke({'instruction': instruction, 'table_definition': table_definition})
    print('create_sql:', create_sql)

    result = spark.sql(create_sql)
    # resultのデータの中身を表示
    print('result:', result)
    print('result_show:')
    result.show()

    df = result.toPandas()
    print('df1:', df)
    for col in df.columns:
        if 'Label' in col:
            df[col] = df[col].astype(str)
    df = df.to_dict(orient="records")
    print('df2:', df)
    return df