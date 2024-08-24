import fitz
import easyocr
import time
import json
import os
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from cnstd import LayoutAnalyzer
from cnocr import CnOcr
import cv2

mat = fitz.Matrix(4, 4)  # high resolution matrix
ocr_time = 0
pix_time = 0

def ocr_batch_imgs(imgs_path):
    img_files = os.listdir(imgs_path)
    images = list()
    for img in img_files:
        if img.endswith(".png"):
            images.append(f"{imgs_path}/{img}")
    ocr = CnOcr(rec_model_name='densenet_lite_136-gru', det_model_name="db_resnet34", rec_model_backend="pytorch", det_model_backend="pytorch")
    res = ocr.ocr_for_single_lines(images, batch_size=10)
    
    print(res)
def get_page(page_path):
    ocr = CnOcr(rec_model_name='densenet_lite_136-gru', det_model_name="db_resnet34", rec_model_backend="pytorch", det_model_backend="pytorch")
    reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
    
    analyzer = LayoutAnalyzer('layout')
    img = cv2.imread(page_path)
    page_layouts = analyzer.analyze(img, resized_shape=704)
    analyzer.save_img(img, page_layouts, page_path)
    page_text = ""
    img_w = img.shape[1]
    right_text = ""
    left_text = ""
    for layout in page_layouts:
        if layout['type'] == 'Text' or layout['type'] == 'Title':
            
            sub_img = img[int(layout['box'][0][1]):int(layout['box'][2][1]), int(layout['box'][0][0]):int(layout['box'][2][0])]
            # text_out = ocr.ocr(sub_img)
            text_out = reader.readtext(sub_img, paragraph=True, text_threshold=0.7, low_text=0.4)
            if int(layout['box'][0][0]) <= 2*img_w // 5:
                for i in range(len(text_out)):
                    text_item = text_out[i]
                    left_text += text_item[1] + "\n"
            else:
                for i in range(len(text_out)):
                    text_item = text_out[i]
                    right_text += text_item[1] + "\n"
                    
            
            # if int(layout['box'][0][0]) <= 2*img_w // 5:
            #     for text_item in text_out:
            #         left_text += text_item['text'] + "\n"
            # else:
            #     for text_item in text_out:
            #         right_text += text_item['text'] + "\n"
    page_text = left_text + right_text
    return page_text

def page2img(page, save_path):
    save_path = save_path.replace(".pdf", '')
    zoom_x = 4.0 # horizontal zoom
    zomm_y = 4.0 # vertical zoom
    mat = fitz.Matrix(zoom_x, zomm_y) # zoom factor 2 in each dimension
    pix = page.get_pixmap(matrix=mat) # use 'mat' instead of the identity matrix

    # pix = page.get_pixmap()  # render page to an image
    if not os.path.exists(save_path): os.makedirs(save_path)
    pix.save(f"{save_path}/{page.number}.png")  # store image as a PNG
    page_text = get_page(f"{save_path}/{page.number}.png")
    return page_text

def load_book(book_path, save_path):
    print(f"load_book:{save_path}")
    doc = fitz.open(book_path)
    ocr_count = 0
    document_text = list()
    pool = ThreadPoolExecutor(max_workers=32)
    job_list = list()
    for page in doc:
        print(f"page:{page.number}")
        # document_text.append(get_page(page))
        # page_img_path = page2img(page, f"{save_path}")
        res = pool.submit(page2img, page, f"{save_path}")
        job_list.append(res)
    
    total = len(job_list)
    fi=0
    for job in as_completed(job_list):
        print(f"job: {fi}/{total}")
        if job.done() and job.result()!=None:
            r = job.result()
            document_text.append(r)
        fi+=1
        
        
    with open(f"{save_path.replace('.pdf', '')}/document.json", 'w', encoding="utf-8") as f:
        json.dump(document_text, f, ensure_ascii=False)
    
def load_all_book(book_path, save_path):
    sub_files = os.listdir(book_path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    for sub_f in sub_files:
        if os.path.isdir(f"{book_path}/{sub_f}"):
            load_all_book(f"{book_path}/{sub_f}", f"{save_path}/{sub_f}")
        else:
            load_book(f"{book_path}/{sub_f}", f"{save_path}/{sub_f}")
            return
            


# with open("docu2text.json","w", encoding="utf-8") as f:
#     json.dump(docu_text, f, ensure_ascii=False)

if __name__=="__main__":
    # load_all_book(book_path="../data/pdf", save_path="../data/load/pdf2img")
    ocr_batch_imgs(imgs_path="../data/load/pdf2img/【医脉通】2017年台湾幽门螺杆菌共识：关于幽门螺杆菌感染的临床管理、筛选治疗和监控以改善台湾地区胃癌控制的共识")
    
# print("-------------------------")
# print("OCR invocations: %i." % ocr_count)
# print(
#     "Pixmap time: %g (avg %g) seconds."
#     % (round(pix_time, 5), round(pix_time / ocr_count, 5))
# )
# print(
#     "OCR time: %g (avg %g) seconds."
#     % (round(ocr_time, 5), round(ocr_time / ocr_count, 5))
# )
