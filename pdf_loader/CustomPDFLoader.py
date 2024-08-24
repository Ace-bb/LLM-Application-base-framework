from conf.setting import *
from conf.prompt import *
from conf.conf import *

from typing import Any, List, Optional
from langchain.docstore.document import Document
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.chains.query_constructor.base import AttributeInfo
import fitz
from LLM.llm import LLM
from pdf_loader.ParseTab import ParseTab, TableLoader


class CustomPDFLoader(PyMuPDFLoader):
    def __init__(self, file_path: str, chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = split_length_function, separators=separators):
        super().__init__(file_path)
        self.file_name = file_path.split('/')[-1]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            length_function = length_function,
            separators = separators
        )
        self.llm = LLM()
        self.table_loader = TableLoader(file_path)
        self.all_page_text = list()
    
    def set_splitter(self, splitter, separator='。\n', chunk_size=1024, chunk_overlap=10):
        self.text_splitter = splitter(chunk_size = chunk_size, chunk_overlap=10)

    def text_split(self, text):
        return self.text_splitter.split_text(text=text)

    def text_clean(self, text):
        # self.llm.setTemplate("下面会给你一段文本，你需要对这段文本进行数据清洗，需要清除掉页眉页脚、作者信息、电话邮箱、英文段落、与主体段落无关的文本，直接返回清洗后的文本，不要有清洗后的文本如下字样。需要清洗的文本为：{text}", ["text"])
        # cleaned_text = self.llm.run({"text": text})
        import jionlp
        cleaned_text = jionlp.clean_text(text)
        return cleaned_text

    def str2list(self, s):
        key_words_list = eval(s)
        if type(key_words_list) == list:
            return key_words_list
        
    def get_text_key_word(self, text):
        self.llm.setTemplate("抽取出以下文本中的关键词，并以数组格式返回：{text}", ["text"])
        key_words = self.llm.run({"text": text})
        try:
            key_words_list = eval(key_words)
        except:
            key_words_list = []

        return key_words_list

    def get_text_conditions(self, text):
        self.llm.setTemplate('''抽取出以下文本中用于判断的条件表达式，表达式参照 ["肿瘤部位为胃"，"cT为T1或T2"]的格式，需要有每个条件的指标、关系和值，以数组格式返回全部条件表达式。文本为：{text}''', ['text'])
        conditions = self.llm.run({"text": text})
        # try:
        #     conditions_list = eval(conditions)
        # except:
        #     conditions_list = []
        return conditions
    
    def check_end_text(self, text:str):
        if '参考文献' in text:
            return text.split('参考文献')[0], True
        else:
            return text, False
    
    def load_ans_save_split_pdf(self, save_path, **kwargs: Optional[Any]):
        """Load file."""

        doc = fitz.open(self.file_path)  # open document
        file_path = self.file_path if self.web_path is None else self.web_path

        documents = list()
        # for i, page in enumerate(doc):
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pagetext = page.get_text(**kwargs)
            pagetext, flag = self.check_end_text(pagetext)
            pagetext = self.text_clean(pagetext)
            text_chunks = self.text_split(pagetext)
            for chunk in text_chunks:
                console.log(f"{chunk}\n\n")
                documents.append(chunk) 
        
        with open(f'{save_path}/{self.file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False)

    def load_and_save_cleaned_pdf(self, save_path, **kwargs: Optional[Any]):
        """Load file."""
        import fitz

        doc = fitz.open(self.file_path)  # open document
        file_path = self.file_path if self.web_path is None else self.web_path

        documents = list()
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pagetext = page.get_text(**kwargs)
            pagetext, flag = self.check_end_text(pagetext)
            text_chunks = self.text_split(pagetext)
            for chunk in text_chunks:
                cleaned_chunk = self.text_clean(chunk)
                documents.append(cleaned_chunk)
        with open(f'{save_path}/{self.file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False)     

    def load_tagged_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            json_file_data = json.load(f)
        documents = list()
        for item in json_file_data:
            documents.append(Document(
                page_content = item['page_content'],
                metadata= item['metadata']
            ))
        return documents
    
    def load_table(self, page_text, page_num):
        if '\n表 'in page_text:
            title_start_pos = page_text.find('\n表')
            title_end_pos = page_text.find('\n', title_start_pos)
            table_name = page_text[title_start_pos:title_end_pos]
            self.table_loader.exist_table([page_num])
            table_data, page_text = self.table_loader.load_table(page_num=[page_num], page_text=page_text)
            return f"{table_name}\n{table_data}", page_text
        elif self.table_loader.exist_table([page_num]):
            table_data, page_text = self.table_loader.load_table(page_num=[page_num], page_text=page_text)
            return table_data, page_text


    def load(self, save=False, save_path=None, **kwargs: Optional[Any]) -> List[Document]:
        if save== True and save_path==None:
            raise "save path 不能为空"
        
        """Load file."""
        import fitz

        doc = fitz.open(self.file_path)  # open document
        file_path = self.file_path if self.web_path is None else self.web_path
        # for row in doc.get_toc():
        #     print(row)
        documents = list()
        save_documents = list()
        # print(f"{doc.page_count}")
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pagetext = page.get_text(**kwargs)
            self.all_page_text.append(pagetext)
            # 读取表格
            # if '\n表 'in pagetext or self.table_loader.exist_table([i+1]):
            #     table, pagetext = self.load_table(page_text=pagetext, page_num=i+1)
            #     documents.append(Document(
            #         page_content=table,
            #         metadata=dict(
            #             {
            #                 "source": file_path,
            #                 "file_path": file_path,
            #                 "page_number": page.number + 1,
            #                 "total_pages": len(doc),
            #                 "tag": '',
            #                 # "conditions": 'conditions',
            #                 "title": doc.metadata['title'],
            #                 "author": doc.metadata['author'],
            #                 "subject": doc.metadata['subject'],
            #                 "keywords": doc.metadata['keywords'],
            #             }
            #         )
            #     ))

            pagetext, flag = self.check_end_text(pagetext)
            # print(f"flag: {flag}")
            pagetext = self.text_clean(pagetext)
            text_chunks = self.text_split(pagetext)
            for j, chunk in enumerate(text_chunks):
                if '�' in chunk: chunk = chunk.replace('�','')
                if len(chunk)<64:continue
                # console.log(f"chunk: {j}, chunk num:{len(text_chunks)}, current page num:{i}, total page num: {len(doc)}")
                # text_tag = self.get_text_key_word(chunk)
                # conditions = self.get_text_conditions(chunk)
                docu = Document(
                    page_content=chunk.replace("\n",""),
                    metadata=dict(
                        {
                            "source": file_path,
                            "file_path": file_path,
                            "page_number": page.number + 1,
                            "total_pages": len(doc),
                            # "tag": text_tag,
                            # "conditions": 'conditions',
                            "title": self.file_name,# doc.metadata['title'],
                            "author": doc.metadata['author'],
                            "subject": doc.metadata['subject'],
                            "keywords": doc.metadata['keywords'],
                        }
                    ),
                )
                if save:
                    save_documents.append(docu.__dict__)
                documents.append(docu)
            if flag: break
        if save and len(save_documents)>0:
            console.log(f"save file: {self.file_name}")
            with open(f'{save_path}/{self.file_name.split(".")[0]}.json', 'w', encoding='utf-8') as f:
                json.dump(save_documents, f, ensure_ascii=False)   
        return documents

    def load_2_str_list(self, save=False, save_path=None, **kwargs: Optional[Any]) -> List[Document]:
        if save== True and save_path==None:
            raise "save path 不能为空"
        
        """Load file."""
        import fitz

        doc = fitz.open(self.file_path)  # open document
        file_path = self.file_path if self.web_path is None else self.web_path
        documents = list()
        save_documents = list()
        print(f"{doc.page_count}")
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pagetext = page.get_text(**kwargs)
            pagetext, flag = self.check_end_text(pagetext)
            # print(f"flag: {flag}")
            pagetext = self.text_clean(pagetext)
            text_chunks = self.text_split(pagetext)
            for j, chunk in enumerate(text_chunks):
                if '�' in chunk: chunk = chunk.replace('�','')
                if len(chunk)<64:continue
                if save:
                    save_documents.append(chunk)
                documents.append(chunk)
            if flag: break
        if save:
            with open(f'{save_path}/{self.file_name}.json', 'w', encoding='utf-8') as f:
                json.dump(save_documents, f, ensure_ascii=False)   
        return documents
    
DOCUMENT_CONTENT_DESCRIPTION = "不同医疗领域，不同疾病的医书、指南或共识书籍"
METADATA_FIELD_INFO=[
    AttributeInfo(
        name="source",
        description="文本的出处", 
        type="string", 
    ),
    AttributeInfo(
        name="file_path",
        description="文件名", 
        type="string", 
    ),
    AttributeInfo(
        name="page_number",
        description="文本所在书中的页数", 
        type="integer", 
    ),
    AttributeInfo(
        name="total_pages",
        description="书籍的总页数", 
        type="integer", 
    ),
    AttributeInfo(
        name="tag",
        description="文本段落中的关键词标签", 
        type="string or list[string]", 
    ),
    AttributeInfo(
        name="conditions",
        description="文本段落中包含的条件", 
        type="string or list[string]", 
    ),
    AttributeInfo(
        name="title",
        description="文章标题",
        type="string"
    ),
    AttributeInfo(
        name="author",
        description="文章作者",
        type="string"
    ),
    AttributeInfo(
        name="subject",
        description="文章所属期刊",
        type="string"
    ),
    AttributeInfo(
        name="keywords",
        description="文章的关键词",
        type="string"
    ),
]