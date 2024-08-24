import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import json
from PIL import Image
import fitz
import numpy as np
from tqdm import tqdm
from paddleocr import PPStructure,draw_structure_result,save_structure_res
# from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
from concurrent.futures import ThreadPoolExecutor, as_completed
from paddleocr import PaddleOCR, draw_ocr
import threading
ocr_lock = threading.Lock()
layout_lock = threading.Lock()
def ocr_text(ocr_engine, img, bbox):
    ocr_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    ocr_lock.acquire()
    result = ocr_engine.ocr(ocr_img, cls=True)
    ocr_lock.release()
    page_content = ""
    if result == None: return ""
    for idx in range(len(result)):
        res = result[idx]
        # for line in res:
        page_content += res[1][0]
    # print(page_content)
    return page_content

def ocr_layout(layout_engine, img):
    # img = cv2.imread(img_path)
    result = layout_engine(img)
    if result == None: return ""
    for line in result:
        line.pop('img')
        print(line)

def start_ocr(ocr_engine, layout_engine, img):
    layout_lock.acquire()
    result = layout_engine(img)
    layout_lock.release()
    if result == None: return ""
    page_contents = list()
    for layout in result:
        print(layout['type']) # header title table_caption figure text footer
        # if layout['type'] == 'text':# or layout['type'] == 'title': 
        #     page_contents.append(ocr_text(ocr_engine, img, layout['bbox']))
    return page_contents

def check_text_enough(ocr_engine, img):
    ocr_lock.acquire()
    result = ocr_engine.ocr(img, cls=True)
    ocr_lock.release()
    page_content = ""
    if result == None: return ""
    for idx in range(len(result)):
        res = result[idx]
        # for line in res:
        page_content += res[1][0]
    # print(page_content)
    return len(page_content)>50

def save_book_img(ocr_engine, layout_engine, page, page_num, save_path, book_name):
    mat = fitz.Matrix(3, 3)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    # if width or height > 2000 pixels, don't enlarge the image
    # if pm.width > 3000 or pm.height > 3000:
    #     pm = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    layout_lock.acquire()
    result = layout_engine(img)
    layout_lock.release()
    if result == None: return
    for layout in result:
        # print(layout)
        x1, y1, x2, y2 = layout['bbox']
        if layout['type'] == 'figure' and (x2-x1)>300 and (y2-y1)>300:# or layout['type'] == 'title':
            if check_text_enough(ocr_engine, img[y1:y2, x1:x2]):
                if not os.path.exists(f"{save_path}"): os.makedirs(f"{save_path}")
                cv2.imwrite(f"{save_path}/{book_name}__{page_num}.png", img[y1:y2, x1:x2])
    

# def page_opration(ocr_engine, layout_engine, page, pg, )
def start_ocr_book(ocr_engine, layout_engine, book_path, save_path:str, book_name):
    document_content = list()
    
    pool = ThreadPoolExecutor(max_workers=128)
    job_list = list()
    with fitz.open(book_path) as pdf:
        for pg in tqdm(range(0, pdf.page_count), desc=f"Add:{book_name}"):
            # print(f"{pg}/{pdf.page_count}")
            try:
                page = pdf[pg]
            except:
                continue
            # mat = fitz.Matrix(3, 3)
            # pm = page.get_pixmap(matrix=mat, alpha=False)
            # # if width or height > 2000 pixels, don't enlarge the image
            # if pm.width > 3000 or pm.height > 3000:
            #     pm = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

            # img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            res = pool.submit(save_book_img, ocr_engine, layout_engine, page, pg, save_path, book_name)
            job_list.append(res)
            # save_book_img(ocr_engine, layout_engine, img, pg, save_path, book_name)
            # document_content.extend(start_ocr(ocr_engine, layout_engine, img))
    total = len(job_list)
    fi=0
    for job in tqdm(as_completed(job_list), total=total, desc=book_path.split("Classify")[-1]):
        # print(f"job: {fi}/{total}")
        fi+=1
    # with open(save_path.replace('.pdf', '.json'), 'w', encoding="utf-8") as f:
    #     json.dump(document_content, f, ensure_ascii=False)
    ...
    
def load_all_books():
    with open("../data/YYYQ/Books.json", "r", encoding="utf-8") as f:
        book_used = json.load(f)
    book_base_path = "/root/nas/projects/Book"
    # table_engine = PPStructure(show_log=False)
    layout_engine = PPStructure(table=False, ocr=False, show_log=False)
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch")
    save_folder = '/root/nas/projects/Book/PaddleOCRBook'
    
    pool = ThreadPoolExecutor(max_workers=64)
    xxx1 = ['耳鼻喉科', '预防保健科', '麻醉科']
    xxx2 = ['外科', '妇产科', '急诊科', '皮肤科', '眼科']
    for folder in xxx1:# book_used.keys()[:5]:
        if folder =="儿科" or folder =="传染科" or folder =="口腔科" or folder =="内科":
            continue  
        for book in book_used[folder]:
            book_file_path = f"{book_base_path}/{book}"
            
            print(book_file_path)
            job_list = list()
            document_content = list()
            with fitz.open(book_file_path) as pdf:
                for pg in range(0, pdf.page_count):
                    print(f"{pg}/{pdf.page_count}")
                    page = pdf[pg]
                    mat = fitz.Matrix(3, 3)
                    pm = page.get_pixmap(matrix=mat, alpha=False)
                    # if width or height > 2000 pixels, don't enlarge the image
                    if pm.width > 3000 or pm.height > 3000:
                        pm = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

                    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    res = pool.submit(start_ocr, ocr_engine, layout_engine, img)
                    job_list.append(res)
                    
        
            total = len(job_list)
            fi=0
            for job in as_completed(job_list):
                print(f"job: {fi}/{total}")
                try:
                    if job.done() and job.result()!=None:
                        r = job.result()
                        document_content.extend(r)
                except Exception as e:
                    print(e)
                fi+=1
            if not os.path.exists(f"{save_folder}/{folder}"): os.makedirs(f"{save_folder}/{folder}")
            with open(f"{save_folder}/{folder}/{book.split('/')[-1].split('.')[0]}.json", 'w', encoding="utf-8") as f:
                json.dump(document_content, f, ensure_ascii=False)
    pool.shutdown()

def load_all_book_v2():
    # classified_book_path = "/root/nas/projects/Book/Guidelines"
    # ocr_book_save_path = "/root/nas/projects/Book/ImageInBooks/Guidelines-Img_page"
    classified_book_path = "/root/nas/projects/Book/Classify"
    ocr_book_save_path = "/root/nas/projects/Book/ImageInBooks/Classify"
    # classified_book_path = "/root/nas/projects/Book/HiQPdfs"
    # ocr_book_save_path = "/root/nas/projects/Book/ImageInBooks/HiQPdfs"
    
    layout_engine = PPStructure(table=False, ocr=False, show_log=False)
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch")
    pool = ThreadPoolExecutor(max_workers=64)
    book_folders = os.listdir(classified_book_path)
    job_list = list()
    pdf_1 = ['儿科', '儿童感染性疾病专家共识或指南2020年至2023年发表中文', '其他', '其他医书']
    pdf_2 = ['内分泌科',  '口腔科',  '呼吸科',  '妇产科']
    pdf_3 = ['影像科',  '心血管科',  '急诊重症科',  '感染科']
    pdf_4 = ['指南1',  '指南2',  '整形外科',  '普通外科']
    pdf_5 = ['精神专科医书',  '翻译',  '解读']
    
    C1 = ['传染科',  '儿科',  '其他',  '内科',  '口腔科',  '外科',  '妇产科',  '急诊科',  '皮肤科',  '眼科',  '耳鼻喉科',  '预防保健科',  '麻醉科']
    for folder in C1[11:]:
        # if folder == "预防保健科": continue
        book_files = os.listdir(f"{classified_book_path}/{folder}")
        
        for book_name in book_files:
            # if not os.path.exists(f"{ocr_book_save_path}/{folder}/{book_name.replace('.pdf', '')}"): os.makedirs(f"{ocr_book_save_path}/{folder}/{book_name.replace('.pdf', '')}")
            # if os.path.exists(f"{ocr_book_save_path}/{folder}/{book_name}"): continue
            print(f"{classified_book_path}/{folder}/{book_name}")
            # start_ocr_book(ocr_engine, layout_engine, f"{classified_book_path}/{folder}/{book_name}", f"{ocr_book_save_path}/{folder}/{book_name}")
            res = pool.submit(start_ocr_book, ocr_engine, layout_engine, 
                        f"{classified_book_path}/{folder}/{book_name}", 
                        f"{ocr_book_save_path}/{folder}",
                        f"{book_name.replace('.pdf', '')}")
            job_list.append(res)
        # break
    
    total = len(job_list)
    fi=0
    for job in as_completed(job_list):
        print(f"job: {fi}/{total}")
        fi+=1

def check_book_exist(search_path, book_name):
    if not os.path.exists(search_path): return False
    book_imgs = os.listdir(search_path)
    for img_name in book_imgs:
        if img_name.split("__")[0] == book_name.replace('.pdf', ''): return True
    return False

def load_all_book_v3():
    # classified_book_path = "/root/nas/projects/Book/Guidelines"
    # ocr_book_save_path = "/root/nas/projects/Book/ImageInBooks/Guidelines-Img_page"
    classified_book_path = "/root/nas/projects/Book/Classify"
    ocr_book_save_path = "/root/nas/projects/Book/ImageInBooks/Classify"
    # classified_book_path = "/root/nas/projects/Book/HiQPdfs"
    # ocr_book_save_path = "/root/nas/projects/Book/ImageInBooks/HiQPdfs"
    
    layout_engine = PPStructure(table=False, ocr=False, show_log=False)
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch")
    book_folders = os.listdir(classified_book_path)
    
    C1 = ['传染科',  '儿科',  '其他',  '内科']
    C2 = [ '口腔科',  '外科',  '妇产科',  '急诊科']
    C3 = ['皮肤科',  '眼科',  '耳鼻喉科']
    C4 = [ '预防保健科',  '麻醉科']
    
    for folder in C1:
        book_files = os.listdir(f"{classified_book_path}/{folder}")
        
        for book_name in book_files:
            # print(book_name)
            if check_book_exist(f"{ocr_book_save_path}/{folder}", book_name): continue
            try:
                start_ocr_book(ocr_engine, layout_engine, 
                            f"{classified_book_path}/{folder}/{book_name}", 
                            f"{ocr_book_save_path}/{folder}",
                            f"{book_name.replace('.pdf', '')}")
            except:
                continue
        # break
    
if __name__=="__main__":
    load_all_book_v3()