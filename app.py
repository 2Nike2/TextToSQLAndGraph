from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from sklearn.datasets import load_iris, load_wine
import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

decide_dataset_prompt = """\
次の指示についてどのデータセットを取得すべきかを決定して下さい。

指示: {instruction}

選択肢：
"iris", "wine"

出力例:
{{"dataset": "target_dataset"}}
"""
decide_dataset_template = PromptTemplate.from_template(decide_dataset_prompt)

decide_graphtype_and_columns_prompt = """\
次の指示と列名一覧から使用すべきグラフ種類と表示する列を下記から選び、JSON形式で出力して下さい。

グラフ種類:
"ScatterChart", "BarChart", "PieChart", "LineChart"

出力例：
{{"graphtype": "ScatterChart", "columns": ["target_column1", "target_column2"]}}

指示: {instruction}
列名: {columns}

"""
decide_graphtype_and_columns_template = PromptTemplate.from_template(decide_graphtype_and_columns_prompt)

decide_graph_option_prompt = """\
次の指示、列名,グラフ種類に基づいて適切なGoogle Chartsのグラフオプションを生成して下さい。

指示: {instruction}
列名: {columns}
グラフ種類: {graphtype}

制約:
・Google Chartsの仕様に厳密に従って下さい。

出力例:
{{
    "title": "GraphTitle",
    "legend": {{"position": "bottom"}}
}}
"""
decide_graph_option_template = PromptTemplate.from_template(decide_graph_option_prompt)

llm = ChatOpenAI()

decide_dataset_chain = decide_dataset_template | llm | JsonOutputParser()

decide_graphtype_and_columns_chain = decide_graphtype_and_columns_template | llm | JsonOutputParser()

decide_graph_option_chain = decide_graph_option_template | llm | JsonOutputParser()

app = Flask(__name__)
CORS(app, resources={"/jsontest": {"origins": " http://127.0.0.1:5500"}})

@app.route('/')
def root_page():
    return redirect(url_for('graph_page'))
    # return 'This is the root page.'

@app.route('/jsontest')
def jsontest():
    print('jsontest')
    return {
        'demand': [i * 10 for i in range(20)]
    }

@app.route('/fetchapitest')
def fetch_page():
    return render_template('fetch_api_test.html')

@app.route('/displaygraph')
def graph_page():
    return render_template('display_graph.html')

@app.route('/getdata', methods=['POST'])
def get_data():
    
    data = request.get_json()
    instruction = data['instruction']
    print(instruction)

    dataset_name = decide_dataset_chain.invoke({'instruction': instruction})['dataset']
    
    if dataset_name == 'iris':
        dataset = load_iris()
    elif dataset_name == 'wine':
        dataset = load_wine()

    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    return df.to_dict(orient="records")

@app.route('/getgraphtypeandcolumns', methods=['POST'])
def get_graphtype_and_columns():
    data = request.get_json()
    instruction = data['instruction']
    columns = data['columns']

    graphtype_and_columns = decide_graphtype_and_columns_chain.invoke(
        {'instruction': instruction, 'columns': columns}
    )

    print('graphtype_and_columns', graphtype_and_columns)

    return graphtype_and_columns

@app.route('/getgraphoption', methods=['POST'])
def get_graph_option():
    data = request.get_json()
    instruction = data['instruction']
    columns = data['columns']
    graphtype = data['graphtype']

    graph_option = decide_graph_option_chain.invoke(
        {'instruction': instruction, 'columns': columns, 'graphtype': graphtype}
    )

    print('graph_option', graph_option)

    return graph_option
    