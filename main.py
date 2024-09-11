from LLM.llm import LLM
from conf.prompt import qa_generate_template
from conf.Tools import Tools
tools = Tools()
from conf.setting import *

import json
import os

os.environ['OPENAI_API_BASE'] = "https://api.xeduapi.com"
os.environ['OPENAI_API_KEY'] = "sk-jz0shLgMJY9HBVnLC3Fe3dCaA5204a418e67003f637f1eFf"
from tqdm import tqdm

function_data = tools.read_json("./datas/genereated_json.json")

llm = LLM()
llm.setTemplate(temp=qa_generate_template, values=["function_intro"])
genereated_qa_json = list()
for item in tqdm(function_data):
    res = llm.run({"function_intro": item})
    try:
        genereated_qa_json.append(json.loads(res.replace('```json', '').replace('```', '')))
    except:
        genereated_qa_json.append(res)

tools.write_2_json(genereated_qa_json, "./datas/genereated_qa_json1.json")