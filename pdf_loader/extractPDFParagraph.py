import os
# os.environ['OPENAI_API_BASE'] = "https://api.emabc.xyz/v1"
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
# api_key = "sk-BFHheN895eZkdU5n54A02b0b581540618f30B748C51e18E7"
# openai.api_key = api_key

separators=['。\n', '\n\n', '。', "\n", " ", ""]
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap  = chunk_overlap,
    length_function = length_function,
    separators = separators
)
character_splitter = CharacterTextSplitter(
    separator = "。\n",
    chunk_size = 500,
    chunk_overlap  = 50,
    length_function = len,
    # is_separator_regex = False,
)


text_splitter = recursive_splitter
@retry(tries=10)
def check_medical(content):
    check_prompt = f"以下这个医学文本是否包含有专业性的医学知识？如果包含有专业性的医学知识则回答True，否则回答False。只需要回答True或False。\n医学问题：{content}\n回答："
    retry_times = 0
    while retry_times<10:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": check_prompt}] 
        )
        message = response["choices"][0]["message"]["content"]
        res = str(message.encode('utf-8').decode('utf-8'))
        if "True" == res: 
            return True
        elif "False" == res:
            return False
        retry_times+=1
    return False

@retry(tries=30)
def generate(prompt, content, title):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )
    message = response["choices"][0]["message"]["content"]
    g_res = message.encode('utf-8').decode('utf-8')
    # check_prompt = f"以下这个问题是否只跟医学相关，并且为医生和患者的对话中可能出现的问题？只需要回答True或False。\n问题：{g_res}"
    # check_res = check_medical([{"role": "user", "content": check_prompt}])
    # if "True" == check_res:
    #     return g_res, content, title
    # else:
    #     return None, content, title
    return g_res, content, title

def check_item_len(para):
    if len(para) < max_len:
        return [para]
    paras = para.split('。')
    res_list = list()
    s = ''
    for p in paras:
        if len(p) > max_len:
            if s != '': res_list.append(s)
            res_list.append(p)
            s = ''
        else:
            s += p
            if len(s) > max_len:
                res_list.append(s)
                s = ''
    return res_list

def add_content(page_text, first_title, second_title):
    chunks = text_splitter.split_text(text=page_text)
    contents = list()
    for item in chunks:
        item = jionlp.clean_text(item.replace('\n','').replace('\n',''))
        # item_res = check_item_len(item)
        # for it in item_res:
        #     if len(it) < 6: continue
        #     it = re.sub("\[.*?\]",'', it)
        #     it = re.sub("\［.*?\］",'', it)
        #     content.append(it)
        #     count_num+=1
        if check_medical(item):
            contents.append({
                "content": item,
                "first_title": first_title,
                "second_title": second_title
            })
    return contents

def merge_content(page_text, first_title, second_title):
    chunks = text_splitter.split_text(text=page_text)
    contents = list()
    for item in chunks:
        # item = jionlp.clean_text(item.replace('\n','').replace('\n',''))
        # contents.append({
        #     "content": item,
        #     "first_title": first_title,
        #     "second_title": second_title
        # })
        contents.append(item)
    return contents

def extract_pdf(base_dir, save_dir, count_num):
    pdfs_list = os.listdir(f'{base_dir}/')
    for pdf_file in pdfs_list:
        doc = fitz.open(f'{base_dir}/{pdf_file}')
        page_num = doc.page_count
        print(f"page_num: {page_num}")
        pdf_contents = list()
        total_page_text = ""
        if not os.path.isdir(f'{save_dir}'):  os.makedirs(f'{save_dir}')
        pdf_filename = f"{pdf_file.split('.')[0]}.json"
        txt_file = f"{pdf_file.split('.')[0]}.txt"
        text_f = open(f"{save_dir}/{txt_file}", 'a', encoding='utf-8')
        for p in range(page_num):
            page = doc.load_page(p)
            page_text = page.get_text("text")
            contents = add_content(page_text, "", "")
            pdf_contents.extend(contents)
            text_f.write(page.get_text("text"))
        # total_text_list = total_page_text.split("。\n")
        # for chunk_text in total_text_list:
        #     print(len(chunk_text))
        #     pdf_content.append(chunk_text.replace("\n", "").replace(" ",""))
        #     count_num+=1
            
        with open(f"{save_dir}/{pdf_filename}", 'w', encoding='utf-8') as f:
            json.dump(pdf_contents, f, ensure_ascii=False) 
        print(f'----------------------total num: {count_num}-------------------------------')

def find_title(page_text):
    first_title = None
    second_title = None
    if page_text.find("####") !=-1:
        first_title_res = re.search(r'####.*####', page_text)
        if first_title_res!=None: first_title = first_title_res.group(0).replace("####", "")
        page_text = re.sub(r'####.*####', "", page_text)
    if page_text.find("###") !=-1:
        second_title_res = re.search(r'###.*###', page_text)
        if second_title_res!=None: second_title = second_title_res.group(0).replace("###", "")
        page_text = re.sub(r'###.*###', "", page_text)
    return page_text, first_title, second_title

def extract_txt(base_dir, save_dir):
    pdfs_list = os.listdir(f'{base_dir}/')
    total_book_contents = list()
    for pdf_file in pdfs_list:
        count_num=0
        with open(f'{base_dir}/{pdf_file}', 'r', encoding='utf-8', newline="") as f:
            doc = f.read().split('。\n')
        page_num = len(doc)
        print(f"page_num: {page_num}")
        txt_contents = list()
        if not os.path.isdir(f'{save_dir}'):  os.makedirs(f'{save_dir}')
        pdf_filename = f"{pdf_file.split('.')[0]}.json"
        txt_file = f"{pdf_file.split('.')[0]}.txt"
        # text_f = open(f"{save_dir}/{txt_file}", 'a', encoding='utf-8')
        first_title = ""
        second_title = ""
        job_list = list()
        for p in range(page_num):
            page_text = doc[p]
            page_text, f_title, s_title = find_title(page_text)
            if f_title!=None: first_title = f_title
            if s_title!=None: second_title = s_title
            # contents = add_content(page_text, first_title, second_title)
            res = pool.submit(add_content, page_text, first_title, second_title)
            job_list.append(res)
        
        total = len(job_list)
        fi=0
        for job in as_completed(job_list):
            print(f"job: {fi}/{total}")
            if job.done() and job.result()!=None:
                r = job.result()
                txt_contents.extend(r)
            fi+=1
            
        with open(f"{save_dir}/{pdf_filename}", 'w', encoding='utf-8') as f:
            json.dump(txt_contents, f, ensure_ascii=False) 
        # with open(f"{save_dir}/{txt_file}", 'w', encoding='utf-8') as f:
        #     f.write(rf"{total_page_text}") 
        # print(f'finish extract: {base_dir}/{folder}/{pdf_file}')
        print(f'----------------------total num: {count_num}-------------------------------')
        # for c in txt_contents:
        #     total_book_contents.append({
        #         "content": c,
        #         "title": pdf_file.split(".")[0]
        #     })
    return "total_book_contents"

# pool = Pool(processes=10)

def add_page_text(page_text, first_title, second_title):
    txt_contents = list()
    page_text = page_text.replace("\n", "").replace(" ", "").replace("\t", "").strip()
    page_text = jionlp.clean_text(page_text)
    if len(page_text)>500:
        page_contents = merge_content(page_text, first_title, second_title)
        for pc in page_contents:
            if check_medical(pc):
                txt_contents.append({
                    "content": pc,
                    "first_title": first_title,
                    "second_title": second_title
                })
    else:
        if check_medical(page_text):
            txt_contents.append({
                "content": page_text,
                "first_title": first_title,
                "second_title": second_title
            })
    return txt_contents
            
def extract_merge_txt(base_dir, save_dir):
    pdfs_list = os.listdir(f'{base_dir}/')
    for pdf_file in pdfs_list:
        with open(f'{base_dir}/{pdf_file}', 'r', encoding='utf-8', newline="") as f:
            doc = f.read().split('。\n')
        page_num = len(doc)
        print(f"page_num: {page_num}")
        txt_contents = list()
        if not os.path.isdir(f'{save_dir}'):  os.makedirs(f'{save_dir}')
        pdf_filename = f"{pdf_file.split('.')[0]}.json"
        txt_file = f"{pdf_file.split('.')[0]}.txt"
        first_title = ""
        second_title = ""
        job_list = list()
        for p in range(page_num):
            page_text = doc[p]
            page_text, f_title, s_title = find_title(page_text)
            if f_title!=None: first_title = f_title
            if s_title!=None: second_title = s_title
            # contents = merge_content(page_text, first_title, second_title)
            res = pool.submit(add_page_text, page_text, first_title, second_title)
            job_list.append(res)
        
        total = len(job_list)
        fi=0
        for job in as_completed(job_list):
            print(f"job: {fi}/{total}")
            if job.done() and job.result()!=None:
                r = job.result()
                txt_contents.extend(r)
            fi+=1
            
        save_txt_contents = list()
        tmp_content = ""
        for content in txt_contents:
            if len(content['content']) >64:
                save_txt_contents.append(content)
                if tmp_content!="":
                    save_txt_contents.append({
                        "content": tmp_content,
                        "first_title": content["first_title"],
                        "second_title": content["second_title"]
                    })
                    tmp_content=""
            else:
                tmp_content+=content['content']
                
        with open(f"{save_dir}/{len(save_txt_contents)}_{pdf_filename}", 'w', encoding='utf-8') as f:
            json.dump(save_txt_contents, f, ensure_ascii=False) 
        print(f'----------------------total num: {len(save_txt_contents)}-------------------------------')
    return "total_book_contents"

def readExcel():
    #1.读取前n行所有数据
    excel_datas=[]
    for sheet in ["体检咨询", "健康问答"]:
        df1=pd.read_excel('../data/Exel/大医重构-知识库构建20230913.xlsx', sheet_name=sheet)#读取xlsx中的第一个sheet
        for i in df1.index.values:
            row_data=df1.loc[i, ['序号','书名','章节','问题','回答']].to_dict()
            content = {
                "Q": list(),
                "content": row_data['回答'],
                "book_name": row_data['书名'],
                "first_title": row_data['章节'],
                "second_title": ""
            }
            excel_datas.append({
                "content": content,
                "title": row_data['书名']
            })
    
    return excel_datas

def start(content):
    check_prompt = f"以下这段文本"
    
def GenerateQA(text_contents, prompt_templates, save_path):
    generate_qas = list()
    job_list = list()
    pool = ThreadPoolExecutor(max_workers=32)
    for text in text_contents:
        # if len(text['content']['content']) < 16: continue
        # llm.setTemplate(temp=prompt, values=['para'])
        content = prompt_templates.format(para=text['content']['content'])
        # res = llm.run({"para": text['content']})
        re_generate = 0
        while re_generate < 5:
            res = pool.submit(generate, [{"role": "user", "content": content}], text['content'], text['title'])
            job_list.append(res)
            re_generate+=1
        # g_res = generate([{"role": "user", "content": content}])
        # print(g_res)
    fi = 0
    total = len(text_contents)*5
    for job in as_completed(job_list):
        print(f"job: {fi}/{total}")
        r = job.result()
        try:
            if job.done() and r[0]!=None:
                generate_qas.append({
                    "Q": r[0],
                    "text": r[1]['content'],
                    "book_name": r[2],
                    "first_title": r[1]['first_title'],
                    "second_title": r[1]['second_title']
                })
        except:
            print(f"result: {r}")
        fi+=1
        
        # print(res)
    pool.shutdown()
    with open(f"{save_path}", 'w', encoding='utf-8') as f:
        json.dump(generate_qas, f, ensure_ascii=False)

@retry(tries=10)
def check_generate_Q(sentence, q_type="D"):
    if "答案" in sentence: return False
    # return True
    doctor_check_prompt = f"要求：1、是一个跟医学相关的专业性的医学题目\n2、只包含一个问题\n3、只有问题，没有包含该问题的答案\n\n你需要检查下面给出的这句话是否符合这3点要求，如果这3点要求全部符合则回答True，否则回答False。只需要回答True或False。\n语句：{sentence}\n回答："
    patient_check_prompt = f"要求：1、是一个患者在问诊过程中可能提出的问题\n2、只包含一个问题!只包含一个问题!只包含一个问题!\n3、必须只有问题，不包含该问题的答案\n\n你需要检查下面给出的这段话是否符合这3点要求，如果这3点要求全部符合则回答True，否则回答False。只需要回答True或False。\n语句：{sentence}\n回答："
    if q_type=="D":
        check_prompt = doctor_check_prompt
    else:
        check_prompt = patient_check_prompt
        
    retry_times = 0
    while retry_times<10:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": check_prompt}] 
        )
        message = response["choices"][0]["message"]["content"]
        res = str(message.encode('utf-8').decode('utf-8'))
        # print(res)
        if "True" == res: 
            return True
        elif "False" == res:
            return False
        retry_times+=1
    return False

def generate_multi_QA(text_content, prompt_templates, q_type="D"):
    content = text_content['content']['content'] #prompt_templates.format(para=text_content['content']['content'])
    results = {
        "Q": list(),
        "content": content,
        "book_name": text_content['title'],
        "first_title": text_content['content']['first_title'],
        "second_title": text_content['content']['second_title']
    }
    
    retry_times = 0
    while retry_times<5:
        prompt = prompt_templates.format(ICL="", para=content, require="")
        # if retry_times==0:
        #     prompt = prompt_templates.format(ICL="", para=content, require="")
        # else:
        #     ICLs = "示例：\n"
        #     for res in results["Q"]:
        #         ICLs += f"{res}\n"
        #     ICLs+="，"
        #     req = "要求生成的问题不得与上面给出的示例重复。"
        #     prompt = prompt_templates.format(ICL=ICLs, para=content, require=req)
        gen_Q = generate([{"role": "user", "content": prompt}], results['content'], results['book_name'])
        try:
            if check_generate_Q(gen_Q, q_type) and gen_Q not in results['Q']:
                results['Q'].append(gen_Q[0])
                retry_times+=1
        except:
            ...
        
        if len(results['Q']) == 5:break
    # if len(results['Q'])==5:
    return results
    # else:
    #     return None
        
def GenerateQA_v2(text_contents, prompt_templates, save_path, q_type="D"):
    generate_qas = list()
    job_list = list()
    pool = ThreadPoolExecutor(max_workers=48)
    for text in text_contents:
        res = pool.submit(generate_multi_QA, text, prompt_templates, q_type)
        job_list.append(res)
            
    fi = 0
    total = len(job_list)
    for job in as_completed(job_list):
        print(f"job: {fi}/{total}")
        r = job.result()
        try:
            if job.done() and r != None:
                generate_qas.append(r)
        except:
            print(f"result: {r}")
        fi+=1
    pool.shutdown()
    with open(f"{save_path}", 'w', encoding='utf-8') as f:
        json.dump(generate_qas, f, ensure_ascii=False)
        
def extractPDFParagraph():
    txt_base_dir = '../data/q_pdf/txts_v2'
    txt_save_dir = '../data/q_pdf/txt_content_v7_check'
    doctor_templates = "{ICL}你是一名医生专家，你需要根据以下医学文本生成一个专业性的医学考试的题目，并且该医学题目的答案就是以下这段医学文本，要求只生成一个问题，不生成答案，并且生成的问题必须是与医学相关的专业性问题。{require}\n医学文本：{para}，医学问题："
    patient_templates = "{ICL}你是一名患者，你需要根据以下医学文本生成一个患者在和医生的对话中可能问到的问题，并且该问题的答案就是以下这段医学文本。要求只生成一个问题，不生成答案，并且生成的问题必须是患者提出的问题。{require}\n医学文本：{para}，患者问题：" 
    # v1 你的症状与以下医学文本中描述的症状相同。现在你需要生成一个你在向医生的问诊中，可能问到的问题，并且这个问题可以用一下文本回答，这个问题要求尽可能的简短
    txt_num = 0
    # total_book_contents = extract_merge_txt(txt_base_dir, txt_save_dir)
    # pdfs_list = os.listdir(f'{txt_save_dir}/')
    # total_book_contents = list()
    # for pdf_file in pdfs_list:
    #     with open(f"{txt_save_dir}/{pdf_file}", 'r', encoding='utf-8') as f:
    #         book_content = json.load(f)
    #         for cont in book_content:
    #             total_book_contents.append({
    #                 "content": cont,
    #                 "title": pdf_file.split(".")[0]
    #             })
    total_book_contents = readExcel()
    # GenerateQA_v2(total_book_contents, doctor_templates, '../data/q_pdf/QA/generate_QAs_excel_doctor_v3.json', q_type="D")
    GenerateQA_v2(total_book_contents, patient_templates, '../data/q_pdf/QA/generate_QAs_excel_patient_v5.json', q_type="P")

def refine_patients_QA():
    data_file = '../data/q_pdf/QA/generate_QAs_patient_v6.json'
    
    refined_datas = []
    with open(f"{data_file}", 'r', encoding='utf-8') as f:
        book_content = json.load(f)
        for c in tqdm(range(len(book_content))):
            content = book_content[c]
            nd = list()
            for i in range(len( content["Q"])):
                content["Q"][i] = content["Q"][i].split("\n")[0]
                if content["Q"][i].startswith("1."):
                    content["Q"][i] = content["Q"][i].replace("1.", "")
                content["Q"][i].strip()
                content["Q"][i].replace(" ", "")
                if "答案" in content["Q"][i]: nd.append(content["Q"][i])
            for s in nd:
                content["Q"].remove(s)
            refined_datas.append(content)
    
    with open("../data/q_pdf/QA/generate_QAs_excel_patient_v4-2.json", 'w', encoding='utf-8') as f:
        json.dump(refined_datas, f, ensure_ascii=False)
    
    #iloc和loc的区别：iloc根据行号来索引，loc根据index来索引。
    #所以1，2，3应该用iloc，4应该有loc

def merge_QA_datas():
    book_data_file = '../data/q_pdf/QA/generate_QAs_doctor_v4.json'
    excel_data_file = "../data/q_pdf/QA/generate_QAs_excel_doctor_v3.json"
    # book_data_file = '../data/q_pdf/QA/generate_QAs_patient_v6-2.json'
    # excel_data_file = "../data/q_pdf/QA/generate_QAs_excel_patient_v4-2.json"
    contents_QAs = list()
    with open(f"{book_data_file}", 'r', encoding='utf-8') as f:
        book_content = json.load(f)
    print(len(book_content))
    
    with open(f"{excel_data_file}", 'r', encoding='utf-8') as f:
        excel_content = json.load(f)
    print(len(excel_content))
    for cont in book_content:
        contents_QAs.append(cont)
    for cont in excel_content:
        contents_QAs.append(cont)
    print(len(contents_QAs))
    with open("../data/q_pdf/QA/全部患者角度的数据.json", 'w', encoding='utf-8') as f:
        json.dump(contents_QAs, f, ensure_ascii=False)
        
if __name__ == '__main__':
    import jionlp
    # base_dir = '../data/q_pdf/pdfs'
    # save_dir = '../data/q_pdf/pdf_content'
    # count_num = 0
    # count_num = extract_pdf(base_dir, save_dir, count_num)
    # txt_base_dir = '../data/q_pdf/txts'
    # txt_save_dir = '../data/q_pdf/txt_content'
    # txt_num = 0
    # total_book_contents = extract_txt(txt_base_dir, txt_save_dir, txt_num)
    # GenerateQA(total_book_contents, '../data/q_pdf/QA')
    # extractPDFParagraph()
    # refine_patients_QA()
    # readExcel()
    merge_QA_datas()