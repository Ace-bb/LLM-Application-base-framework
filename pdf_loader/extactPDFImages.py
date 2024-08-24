# 打开pdf读取文本和图片内容
# pip install PyMuPDF

import os
import sys
import fitz

if __name__ == '__main__':

    # print(sys.argv)

    pdf_document_path = "/home/PJLAB/libinbin/Dev/Project/Decision-Tree/data/pdf/The Patient History An Evidence-Based Approach to Differential Diagnosis by Mark C. Henderson, Lawrence M. Tierney Jr., Gerald W. Smetana (z-lib.org).pdf"
    save_path = "/home/PJLAB/libinbin/Dev/Project/Decision-Tree/data/pdf_imgs"
    # 打开pdf文件
    doc = fitz.open(pdf_document_path)

    # 获取pdf信息
    # print(doc.metadata)  

    # 获取pdf页数
    pages_count = doc.page_count
    print(f'"{pdf_document_path}"总共{pages_count}页')

    # images save path
    images_save_prefix = os.path.abspath(pdf_document_path)[:-4]

    # 逐页读取数据
    for i in range(pages_count):

        # 读取PDF第i页
        # print(f"开始读取第{i + 1}页")
        page = doc.load_page(i) 

        # 获取文本信息
        page_text = page.get_text("text") 
        # print(page_text)

        # 获取图片信息
        page_images = doc.get_page_images(i)

        page_img_idx = 0
        # 转存图片数据
        for image in page_images:
            # load img data
            xref = image[0]
            pix = fitz.Pixmap(doc, xref)
            # image save path
            page_img_idx += 1
            img_save_path = f"{save_path}/page{i}_{page_img_idx}.jpg"
            # save
            # if str(fitz.csRGB) == str(pix.colorspace):
            #     print(f"save: {img_save_path}")
            #     pix.save(img_save_path)
            try:
                if pix.n < 5:       
                    # GRAY or RGB
                    pix.save(img_save_path)
                else:               
                    # CMYK
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    pix1.save(img_save_path)
                    pix1 = None
            except:
                print(str(pix.colorspace))
            pix = None

    print('执行完成！')

