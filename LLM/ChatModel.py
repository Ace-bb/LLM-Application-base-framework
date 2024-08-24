
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

class ChatModel:
    def __init__(self) -> None:
        self.chat = None
        self.messages = list()
        
    def init_chat(self):
        global api_key_id
        self.open_api_keys = OPENAI_API_KEYS
        self.api_key_use = random.sample(self.open_api_keys, 1)[0]
        if ONE_API_KEY!=None: self.api_key_use = ONE_API_KEY
        # console.rule(f"[bold red]{self.api_key_use}")
        # console.log(f"[bold red]ChatModel:{api_key}")
        self.chat = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=self.api_key_use, max_retries=20)
    
    def init_messages(self, messages):
        tmp_msgs = list()
        for msg in messages:
            if msg['type'] == 'system':
                tmp_msgs.append(SystemMessage(content=msg['content']))
            elif msg['type'] == 'human':
                tmp_msgs.append(HumanMessage(content=msg['content']))
            else:
                tmp_msgs.append(AIMessage(content=msg['content']))
        self.messages = tmp_msgs
        return tmp_msgs
        
    def generate(self, messages):
        self.init_chat()
        msgs =  self.init_messages(messages)
        ai_msg = self.chat(msgs)
        return ai_msg.content