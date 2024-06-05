from flask import Flask, render_template,request
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

decide_graph_option_prompt = """\
次の指示、列名に基づいて適切なEChartsのグラフオプションを生成して下さい。

指示: {instruction}
列名: {columns}

制約:
・データの配列が必要なところは「data["列名"]」のように指定して下さい。

出力例:
{{
  xAxis: {{}},
  yAxis: {{}},
  series: [
    {{
      symbolSize: 20,
      data: data["target_column1", "target_column2"],
      type: "scatter"
    }}  
  ]
}}
"""
decide_graph_option_template = PromptTemplate.from_template(decide_graph_option_prompt)

llm = ChatOpenAI()

decide_dataset_chain = decide_dataset_template | llm | JsonOutputParser()

decide_graph_option_chain = decide_graph_option_template | llm | JsonOutputParser()

app = Flask(__name__)
CORS(app, resources={"/jsontest": {"origins": " http://127.0.0.1:5500"}})

@app.route('/')
def root_page():
    return 'This is the root page.'

@app.route('/jsontest')
def jsontest():
    print('jsontest')
    return {
        'demand': [i * 10 for i in range(20)]
    }

@app.route('/fetchapitest')
def fetch_page():
    return render_template('fetch_api_test.html')

@app.route('/showgraph')
def graph_page():
    return render_template('show_graph.html')

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

    return df.to_json()
   
@app.route('/getgraphoption', methods=['POST'])
def get_graph_option():
    data = request.get_json()
    instruction = data['instruction']
    columns = data['columns']

    graph_option = decide_graph_option_chain.invoke({'instruction': instruction, 'columns': columns})

    print('graph_option', graph_option)

    return {'test': 1}
    