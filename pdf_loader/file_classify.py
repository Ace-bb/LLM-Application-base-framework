from LLM.llm import LLM
import os
import json
from retry import retry
import shutil

# @retry(tries=30)
def classify(file_name):
    llm = LLM()
    llm.setTemplate(temp="医院的科室有：急诊科、内科、外科、妇产科、儿科、眼科、耳鼻喉科、口腔科、皮肤科、麻醉科、传染科、预防保健科。下面给你一本书的书名，请你对这本书所属的科室进行划分。要求每本书只划分到一个科室，并且科室只能为前面给出的这几种科室中的一个，如果这本书不属于上面科室中的任何一个，则分为其他。医书名称为：{book}，所属科室为：", values=['book'])
    retry_times = 0
    while retry_times<20:
        Department = llm.run({"book": file_name})
        if Department in ["急诊科", "内科", "外科", "妇产科", "儿科", "眼科", "耳鼻喉科", "口腔科", "皮肤科", "麻醉科", "传染科", "预防保健科", "其他"]:
            return Department
        else:
            retry_times+=1
            # raise ValueError(f"划分科室错误:{Department}")
    return "其他"

def check_file_exist(save_path, file_name):
    sub_files = os.listdir(save_path)
    for sub_f in sub_files:
        if os.path.isdir(f"{save_path}/{sub_f}"):
            res = check_file_exist(f"{save_path}/{sub_f}", file_name)
            if res==True: return True
        else:
            if sub_f == file_name:
                return True
    return False

def classify_files(file_path, file_name):
    save_path="/root/nas/projects/Book/Classify"
    if check_file_exist(save_path, file_name): return
    # if os.path.exists(f"{save_path}/{department}/{file_name}"): return 
    department = classify(file_name)
    
    if not os.path.exists(f"{save_path}/{department}"): os.makedirs(f"{save_path}/{department}")
    print(f"{department}----{file_path}")
    shutil.copy(file_path, f"{save_path}/{department}")
            

def search_folder(folder_path):
    sub_files = os.listdir(folder_path)
    for sub_f in sub_files:
        if os.path.isdir(f"{folder_path}/{sub_f}"):
            search_folder(f"{folder_path}/{sub_f}")
        else:
            if sub_f.endswith(".pdf"):
                classify_files(f"{folder_path}/{sub_f}", sub_f)

def remove_animal_books(folder_path):
    folder_files = os.listdir(folder_path)
    for folder in folder_files:
        files = os.listdir(f"{folder_path}/{folder}")
        for f in files:
            if "犬" in f or "猫" in f or "动物" in f or "禽" in f or "猪" in f or "兽" in f or "羊" in f: # "牛" in f or 
                print(f"{folder_path}/{folder}/{f}")
                os.remove(f"{folder_path}/{folder}/{f}")
                # break

def move_txt_books(folder_path):
    folder_files = os.listdir(folder_path)
    move_path = "/root/nas/projects/Book/TxtBooks/"
    for folder in folder_files:
        files = os.listdir(f"{folder_path}/{folder}")
        for f in files:
            if ".txt" in f: # "牛" in f or 
                print(f"{folder_path}/{folder}/{f}")
                try:
                    shutil.move(f"{folder_path}/{folder}/{f}", move_path)
                except:
                    ...

def move_different_type_books(folder_path):
    folders = os.listdir(folder_path)
    type_sets = set()
    for f in folders:
        if os.path.isdir(f"{folder_path}/{f}"): 
            move_different_type_books(f"{folder_path}/{f}")
        else: #or ".json" in folder
            try:
                if f.endswith(".docx") or f.endswith(".doc"):
                    shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/Docx")
                elif f.endswith(".ppt") or f.endswith(".pptx") or f.endswith(".PPTX"):
                    shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/ppt")
                elif f.endswith(".mobi"):
                    shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/mobi")
                elif f.endswith(".epub"):
                    shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/epub")
                elif f.endswith(".rar") or f.endswith(".zip"):
                    shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/rar_zip")
                elif f.endswith(".djvu"):
                    shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/djvu")
                elif f.endswith(".ebk3"):
                    shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/ebk3")
                elif f.endswith(".chm"):
                    shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/chm")
                elif f.endswith(".xls"):
                    shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/xls")
                else:
                    if not f.endswith(".pdf") and not f.endswith(".PDF") and not f.endswith(".txt"):
                        shutil.move(f"{folder_path}/{f}", "/root/nas/projects/Book/OtherTypeBooks/else")
            except:
                ...
                
            
if __name__=="__main__":
    # search_folder("/root/nas/projects/Book/Books2")
    # remove_animal_books("/root/nas/projects/Book/OCR-Classified")
    move_different_type_books("/root/nas/projects/Book/Books")
    move_different_type_books("/root/nas/projects/Book/Books2")
    move_different_type_books("/root/nas/projects/Book/Hosipital")
    move_different_type_books("/root/nas/projects/Book/Hosipital2")
            