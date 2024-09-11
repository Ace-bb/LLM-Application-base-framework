from conf.setting import *
from conf.prompt import *

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import openai
from openai.error import RateLimitError
from langchain.chat_models import ChatOpenAI
import random
# os.environ['OPENAI_API_BASE'] = "https://fast.xeduapi.com/v1"
# # os.environ['OPENAI_API_BASE'] = "https://api.xeduapi.com"
# ONE_API_KEY = "sk-jz0shLgMJY9HBVnLC3Fe3dCaA5204a418e67003f637f1eFf"
# model_name_base = 'gpt-3.5-turbo-0125'
# model_name_16k = 'gpt-3.5-turbo-16k'
# # gpt_4_model = "gpt-4"
# gpt_4_model = "gpt-4-0125-preview"
# model_name = gpt_4_model
class LLM:
    def __init__(self,model_name=model_name) -> None:
        self.llm = None #ChatOpenAI(model_name=model_name, max_retries=max_retries_times, openai_api_key=None)
        # self.llm = HuggingFacePipeline.from_model_id(model_id='bigscience/bloomz-560m', task="text-generation", device=0)
        self.template =  None
        self.prompt = None# PromptTemplate(template=self.template, input_variables=["text","node"])
        self.llm_chain = None# LLMChain(prompt=self.prompt, llm=self.llm)
        self.open_api_keys = list()
        self.api_key_length = -1
        self.llm_lock = threading.Lock()
        self.api_key_use = None

    def init_llm(self):
        self.api_key_use = ONE_API_KEY
        # console.rule(f"[bold red]{self.api_key_use}")
        # console.log(os.environ['OPENAI_API_BASE'])
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=self.api_key_use, openai_api_base="https://fast.xeduapi.com/v1", max_retries=max_retries_times, temperature=1.2)
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def get_llm(self):
        self.init_llm()
        return self.llm
    
    def setTemplate(self, temp=None, values:list = None):
        if temp!=None: self.template = temp
        self.prompt = PromptTemplate(template=self.template, input_variables=values)
        # self.init_llm()
        
    def run(self, datas):
        self.llm_lock.acquire()
        self.init_llm()
        try_num = 0
        while try_num<200:
            try:
                res = self.llm_chain.run(datas)
                self.llm_lock.release()
                # console.log(f"[bold red]data:{datas}--res: {res}")
                return res
            except RateLimitError:
                console.log(f"[bold red]{self.api_key_use} limit!!!")
                self.init_llm()
                try_num+=1
            except Exception as e:
                # console.log(e)
                self.init_llm()
                try_num+=1
        self.llm_lock.release()
        return None
    
    def run_with_prompt(self, prompt, paras, data):
        self.llm_lock.acquire()
        self.setTemplate(temp=prompt, values=paras)
        self.init_llm()
        try_num = 0
        while try_num<20:
            try:
                res = self.llm_chain.run(data)
                self.llm_lock.release()
                return res
            except RateLimitError:
                console.log(f"[bold red]{self.api_key_use} limit!!!")
                self.init_llm()
                try_num+=1
            except Exception as e:
                self.init_llm()
                try_num+=1
        self.llm_lock.release()
        
    @retry(tries=20)
    def check_yes_or_no(self, prompt_temp, data):
        self.prompt = prompt_temp
        self.init_llm()
        res = self.llm_chain.run(data)
        if str(res)=='True': return True
        elif str(res)=="False": return False
        else: raise ValueError("LLM 生成的结果不为true或false")
    
    def bool_check(self, datas)-> bool:
        self.init_llm()
        try_num = 0
        while try_num<20:
            try:
                res = self.llm_chain.run(datas)
                if 'True' in str(res): return True
                elif "False" in str(res): return False
                else: try_num+=1 
            except RateLimitError:
                console.log(f"[bold red]{openai.api_key} limit!!!")
                self.init_llm()
                try_num+=1
            except: # openai.error.AuthenticationError
                try_num+=1
        return False
    
    def multi_bool_check(self, datas, check_times = 5):
        check_res = {
            "True": 0,
            "False": 0
        }
        for i in range(check_times):
            res = self.bool_check(datas)
            if res:
                check_res['True']+=1
            else:
                check_res['False']+=1
        if max(check_res.keys(), key=lambda item: check_res[item])=="True": return True
        else: return False
                    
    @retry(tries=20)
    def generate(self, prompt_temp, data):
        self.prompt = prompt_temp
        self.init_llm()
        res = self.llm_chain.run(data)
        return res
    
    @retry(tries=20)
    def generate_new_node(self, prompt_temp, data):
        if len(data['label'])==0: return data['condition']
        self.prompt = prompt_temp
        self.init_llm()
        res = self.llm_chain.run(data)
        return res
if __name__ == "__main__":
    llm = LLM()
    text = '''
    2.分期诊断
胃癌的分期诊断主要目的是在制订治疗方案之前充分
了解疾病的严重程度及特点，以便为选择合理的治疗模式提


供充分的依据。胃癌的严重程度可集中体现在局部浸润深度、
淋巴结转移程度以及远处转移存在与否 3 个方面，在临床工
作中应选择合适的辅助检查方法以期获得更为准确的分期胃癌的病理报告应包括与患者治疗和预后相关的所有
内容，如标本类型、肿瘤部位、大体分型、大小及数目、组
织学类型、亚型及分级、浸润深度、脉管和神经侵犯、周围胃癌治疗的总体策略是以外科为主的综合治疗，为进一
步规范我国胃癌诊疗行为，提高医疗机构胃癌诊疗水平，改
善胃癌患者预后，保障医疗质量和医疗安全，特制定本指南。
本指南所称的胃癌是指胃腺癌，包括胃食
管结合部癌。
二、诊断
应当结合患者的临床表现、内镜及组织病理学、影像学
检查等进行胃癌的诊断和鉴别诊断。
临床表现本文主要介绍胃癌的TNM分期、组织学类型和分级、大体分型、病理学报告标准模板、影像学报告指南、影像诊断流程、淋巴结分组标准、胃肿瘤的解剖部位编码、胃食管结合部示意图、胃食管结合部肿瘤Siewert分 型、CT分期征象及报告参考、超声内镜分期征象、常用系统治疗方案、常用靶向治疗药物、放射及化学治疗疗效判定基本标准和肿瘤术前辅助治疗疗效评估。
    '''
    node = {
        "节点类型": "条件节点",
        "节点描述": "非转移性胃癌",
        "判断条件": "肿瘤部位=='胃癌' and cM==M0"
    }

    prompt = '''
    给你一段医学文本和一颗有if-else组成的决策树，你需要根据医学文本补充决策树中的None部分，None可以是下一个if条件判断，也可以是最终的治疗方案
    医学文本为：{text}，决策树为：{tree}
    '''

    tree = '''
if 胃癌综合治疗
    if ( 肿瘤部位 不包含 '食管' )
        if ( cM 为 'M1' )
            None
        else
            None
    else
        if ( cM 为 'M0' )
            None
        else
            None
else
    if ( 肿瘤部位 包含 '食管' )
        None
    else'''
    llm.setTemplate(temp=prompt, values=['text', 'tree'])
    res = llm.run({'text':text, "tree":tree})
    print(res)
