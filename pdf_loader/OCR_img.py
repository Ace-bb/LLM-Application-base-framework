

from cnocr import CnOcr
import cv2
import numpy as np
import os
import easyocr
from retry import retry
from LLM.llm import LLM
ocr = CnOcr(rec_model_name='densenet_lite_136-gru', det_model_name="db_resnet34", rec_model_backend="pytorch", det_model_backend="pytorch")
reader = easyocr.Reader(['ch_sim','en'])

@retry(tries=20)
def gpt_clean(text):
    llm = LLM()
    llm.setTemplate(temp="下面给你一段OCR识别的患者报告单，你需要对识别的文本进行数据清洗。首先需要去除掉与患者报告无关的文字，并纠正报告中识别错误的信息。返回清洗干净的文本。OCR识别的患者报告单为：{text}，清洗后的结果为：")
    return llm.run({"text": text})

def ocr_img(img_path:str):
    if os.path.isdir(img_path):
        files = os.listdir(img_path)
        for f in files:
            ocr_img(f"{img_path}/{f}")
    elif img_path.endswith(".jpg") or img_path.endswith(".png"):
        ocr_result = reader.readtext(img_path, paragraph=True)
        img_text = "\n\n".join([re[1] for re in ocr_result])
        clean_text = gpt_clean(img_text)
        with open(f'{img_path.split(".")[0]}.txt', 'w', encoding='utf-8') as f:
            f.write(clean_text)

if __name__=="__main__":
    ocr_img("/root/nas/projects/PICU/重症感染20230829/02-上传图片")