import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import json
from PIL import Image
from paddleocr import PPStructure,draw_structure_result,save_structure_res
# from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
from concurrent.futures import ThreadPoolExecutor, as_completed
from paddleocr import PaddleOCR, draw_ocr

def save_bbox_img(img_path, result, save_path):
    font_path = '/root/nas/projects/PaddleOCR/doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
    image = Image.open(img_path).convert('RGB')
    im_show = draw_structure_result(image, result,font_path=font_path)
    im_show = Image.fromarray(im_show)
    img_name = img_path.split("/")[-1].split(".")[0]
    im_show.save(f'{save_path}/{img_name}.jpg')

def ocr_text(ocr_engine, img_path, save_path):
    print(img_path)
    result = ocr_engine.ocr(img_path, cls=True)
    page_content = ""
    for idx in range(len(result)):
        res = result[idx]
        # for line in res:
        print(res)
        print()

def ocr_layout(layout_engine, img_path, save_path):
    img = cv2.imread(img_path)
    result = layout_engine(img)
    save_structure_res(result, save_path, os.path.basename(img_path).split('.')[0])

    for line in result:
        line.pop('img')
        print(line)
        
def pp_ocr_img(ocr_engine, img_path, save_path):
    print(img_path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    img = cv2.imread(img_path)
    result = ocr_engine(img)
    save_structure_res(result, save_path, os.path.basename(img_path).split('.')[0])
    page_content = ""
    for line in result:
        line.pop('img')
        if line['type'] == "text" or line['type'] == 'title':
            text_list = line['res']
            line_content = ""
            for text in text_list:
                line_content += text['text']
            page_content+= line_content + '\n'
    save_bbox_img(img_path, result, save_path)
    print(page_content)
    return page_content
    
# img_path = '../data/load/pdf2img/【医脉通】胃癌卵巢转移诊断和治疗中国专家共识（2021版）/1.png'

# def load_pdf_by_paddle()
def load_all_imgs(img_save_path):
    table_engine = PPStructure(show_log=False)
    layout_engine = PPStructure(table=False, ocr=False, show_log=True)
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch")
    save_folder = '../data/load/paddleocr'
    document_text = list()
    pool = ThreadPoolExecutor(max_workers=12)
    folders = os.listdir(img_save_path)
    for folder in folders:
        img_files = os.listdir(f"{img_save_path}/{folder}")
        job_list = list()
        for img_f in img_files:
            # ocr_text(ocr_engine, f"{img_save_path}/{folder}/{img_f}", f"{save_folder}/{folder}")
            ocr_layout(layout_engine, f"{img_save_path}/{folder}/{img_f}", f"{save_folder}/{folder}")
            print(f"---------------------------{img_f}------------------------")
            # pp_ocr_img(ocr_engine=table_engine, img_path=f"{img_save_path}/{folder}/{img_f}", save_path=f"{save_folder}/{folder}")
            # res = pool.submit(pp_ocr_img, table_engine, f"{img_save_path}/{folder}/{img_f}", f"{save_folder}/{folder}")
            # job_list.append(res)
        print()
        
        return
        total = len(job_list)
        fi=0
        for job in as_completed(job_list):
            print(f"job: {fi}/{total}")
            if job.done() and job.result()!=None:
                r = job.result()
                document_text.append(r)
            fi+=1

        with open(f"{save_folder}/{folder}.json", 'w', encoding="utf-8") as f:
            json.dump(document_text, f, ensure_ascii=False)

    # print(line['type'])

if __name__=="__main__":
    load_all_imgs("../data/load/pdf2img")

# h, w, _ = img.shape
# res = sorted_layout_boxes(result, w)
# convert_info_docx(img, res, save_folder, os.path.basename(img_path).split('.')[0])