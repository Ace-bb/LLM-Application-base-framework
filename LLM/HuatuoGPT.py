
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


# print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)
# tokenizer = AutoTokenizer.from_pretrained("FreedomIntelligence/HuatuoGPT2-13B", use_fast=True, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("FreedomIntelligence/HuatuoGPT2-13B", device_map="auto", torch_dtype='auto', trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained("FreedomIntelligence/HuatuoGPT2-13B")
# messages = []
# messages.append({"role": "user", "content": "肚子疼怎么办？"})
# response = model.HuatuoChat(tokenizer, messages)
# print(response)

class HuatuoGPT:
    def __init__(self, model_name, gpu_divice:int = 0) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        self.model = None
        self.llm_lock = None
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype='auto', trust_remote_code=True)
        self.init_one_llm(model_name)
        
        
    def init_one_llm(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype='auto', trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.llm_lock = threading.Lock()
        
    
    def setTemplate(self, temp, values:list):
        self.template = temp
        
    def generate(self, messages, history=[]):
        response, history = self.model.chat(self.tokenizer, messages, history)
        return response, history
    
    def run(self, question):
        msg = self.template.format(**question)
        # messages = self.template.format(**question)
        messages=[]
        messages.append({"role": "user", "content": msg})
        self.llm_lock.acquire()
        response = self.model.HuatuoChat(self.tokenizer, messages)
        self.llm_lock.release()
        return response

    
    