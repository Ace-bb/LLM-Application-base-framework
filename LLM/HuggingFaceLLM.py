
from conf.setting import *
from conf.prompt import *
from conf.conf import *

from conf.setting import api_key_id
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import openai
from openai.error import RateLimitError
from utils.utils import generate
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from transformers import AutoTokenizer, AutoModel
import threading
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from torch import Tensor, device


# print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)

class HuggingFaceLLM:
    def __init__(self, model_name, gpu_divice:int = 0, multi_run:bool = False) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = None
        self.tokenizer_list = list()
        self.model_list = list()
        self.semaphore = threading.Semaphore(7)
        if multi_run:
            self.init_llm(model_name)
        else:
            self.init_one_llm(model_name)
        # self.model = None
        # self.llm = ChatOpenAI(model_name='gpt-3.5-turbo')
#         self.llm = HuggingFacePipeline.from_model_id(model_id=model_name, task="text-generation", device=gpu_divice, batch_size=4)
#         self.template =  """问题: {question}

# 回答: """
#         self.prompt = PromptTemplate(template=self.template, input_variables=["question"])
#         self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
    
    def init_one_llm(self, model_name):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
        self.llm_lock = threading.Lock()
        
    def init_llm(self, model_name):
        for i in range(7):
            print(i)
            # tmp_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tmp_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda(device(f"cuda:{i}"))
            tmp_model = tmp_model.eval()
            # self.tokenizer_list.append(tmp_tokenizer)
            self.model_list.append(tmp_model)
    
    def setTemplate(self, temp, values:list):
        self.template = temp
        # self.prompt = PromptTemplate(template=self.template, input_variables=values)
        # self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        
    def generate(self, messages, history=[]):
        response, history = self.model.chat(self.tokenizer, messages, history)
        return response, history
    
    def run(self, question):
        messages = self.template.format(**question)
        self.llm_lock.acquire()
        response, history = self.model.chat(self.tokenizer, messages, [])
        self.llm_lock.release()
        return response
        # return self.llm_chain.run(question)

    def multi_run(self, question):
        messages = self.template.format(**question)
        self.semaphore.acquire()
        run_model = self.model_list.pop()
        # run_tokenizer = self.tokenizer_list.pop()
        response, history = run_model.chat(self.tokenizer, messages, [])
        self.model_list.append(run_model)
        # self.tokenizer_list.append(run_tokenizer)
        self.semaphore.release()
        return response
    
class LangChainIndicator:
    def __init__(self, model_name = 'bigscience/bloomz-560m', gpu_divice:int = 0) -> None:
        # self.llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        self.llm = HuggingFacePipeline.from_model_id(model_id=model_name, task="text-generation", device=gpu_divice, batch_size=4)
        self.template =  """问题: {question}

回答: """
        self.prompt = PromptTemplate(template=self.template, input_variables=["question"])
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
    
    def setTemplate(self, temp, values:list):
        self.template = temp
        self.prompt = PromptTemplate(template=self.template, input_variables=values)
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
    
    def run(self, question):
        return self.llm_chain.run(question)
    