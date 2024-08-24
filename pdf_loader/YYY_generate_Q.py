import os
import re
import fitz
import openai
from retry import retry
import json
import re
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
chunk_size = 300
chunk_overlap = 50
length_function = len
max_len = chunk_size
import numpy as np
from multiprocessing.dummy import Pool as Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from LLM.llm import LLM

def generate_Q(content):
    llm = LLM()
    llm.setTemplate(temp="你是一个知识问答的问题生成器，下面给你一段医学文本，你需要以这段文本为答案生成一个专业性的医学问题。要求只生成一个问题。 医学文本：{content}，生成的专业性问题为：", values=["content"])
    t = 1
    while t<10:
        try:
            Q = llm.run({"content": content})
            if "\n" in Q: return None
            else: return Q
        except:
            t+=1
            
def summary_content_knowledge_v2(content):
    llm = LLM() 
    llm.setTemplate(temp="下面给你一段医书中的文本，请你总结出这段文本中包含的医学知识的定义。要求按照什么是什么的格式进行总结，并且仅生成一个医学名词的定义。如果文本中不包含可以总结的医学名词，则返回“None”文本：{content}，总结的知识为：", values=["content"])
    t = 1
    while t<10:
        try:
            res = llm.run({"content": content})
            if "None" not in res:
                return res
            else: return None
        except:
            t+=1


def summary_content_knowledge(content):
    llm = LLM()
    llm.setTemplate(temp="下面给你一段医书中的文本，请你总结出这段文本中包含的医学知识。文本：{content}，总结的知识为：", values=["content"])
    t = 1
    while t<10:
        try:
            res = llm.run({"content": content})
            return res
        except:
            t+=1
            
def check_medical_relate(content):
    llm = LLM()
    llm.setTemplate(temp="下面给你一段文本，请你判断这段文本是否包含有疾病相关的知识。如果这段文本讲述的是疾病的相关术语、科学知识、病原学、各种诊断方法、各种检查、评估方法、治疗方法、用药指南和治疗方案相关的内容，则判定这段文本包含有重要的医学知识，返回True。否则，文本与这些方面的知识都无关，则返回False。文本：{content}，是否包含有疾病相关的知识：", values=["content"])
    t = 1
    while t<10:
        try:
            res = llm.run({"content": content})
            if str(res) == "True":
                return True
            elif str(res) == "False":
                return False
            else: t+=1
        except:
            t+=1

def check_QA(Q, A):
    llm = LLM()
    llm.setTemplate(temp="下面给你一个题目和一段文本，你需要判断这段文本可不可以作为这个题目的答案，要求这段文本回答了这个题目的问题，并且这段文本不包含与题目无关的内容。如果可以，返回True，否则返回False。问题为：{Q}，文本为：{A}，文本能否作为A的答案：", values=["Q", "A"])
    t = 1
    while t<10:
        try:
            res = llm.run({"Q": Q, "A":A})
            if str(res) == "True":
                return True
            elif str(res) == "False":
                return False
            else: t+=1
        except:
            t+=1
            
def start_generate(content):
    if not check_medical_relate(content): return None
    knowledge_content = summary_content_knowledge(content)
    if knowledge_content == None: return None
    Q = generate_Q(knowledge_content)
    if Q==None: return None
    if check_QA(Q, knowledge_content):
        return {
                "Q": Q,
                "A": knowledge_content
            }
    else: return None
    
def load_book_and_generate():
    book_save_path = "/root/nas/projects/Book/PaddleOCRBook"
    QA_save_path = "data/YYYQ/v2"
    folders = os.listdir(book_save_path)
    for folder in folders:
        if not os.path.exists(f"{QA_save_path}/{folder}"): os.makedirs(f"{QA_save_path}/{folder}")
        book_files = os.listdir(f"{book_save_path}/{folder}")
        
        for book_name in book_files:
            if os.path.exists(f"{QA_save_path}/{folder}/{book_name}"): continue
            book_content = ""
            print(f"{book_save_path}/{folder}/{book_name}")
            with open(f"{book_save_path}/{folder}/{book_name}", 'r', encoding="utf-8") as f:
                book_content = json.load(f)
            Books_Q_Answer=list()
            book_content = list(filter(lambda item: len(item)>64 or len(item)<2048, book_content))
            print(len(book_content))
            pool = ThreadPoolExecutor(max_workers=42)
            job_list = list()
            if len(book_content) > 400:
                random_book_content = np.random.choice(book_content, size=400, replace=False)
            else:
                random_book_content = book_content
            for i, content in enumerate(random_book_content):
                print(f"{i}/{len(random_book_content)}")
                
                job = pool.submit(start_generate, content)
                job_list.append(job)
                
            fi = 0
            total = len(job_list)
            for job in as_completed(job_list):
                print(f"job: {fi}/{total}")
                r = job.result()
                try:
                    if job.done() and r != None:
                        Books_Q_Answer.append(r)
                except:
                    print(f"result: {r}")
                fi+=1
            pool.shutdown()
    
                # if len(Books_Q_Answer) > 200: break
                
            with open(f"{QA_save_path}/{folder}/{book_name}", 'w', encoding="utf-8") as f:
                json.dump(Books_Q_Answer, f, ensure_ascii=False)