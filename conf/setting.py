from conf.packages import *
from conf.conf import *
from conf.Tools import Tools
from utils import *
tools = Tools()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

os.environ['OPENAI_API_BASE'] = ""
# os.environ['OPENAI_API_BASE'] = ""
ONE_API_KEY = "sk-"

max_retries_times = 30

console = Console(record=False, soft_wrap=True)


# 超参
similarity_search_num = 1
key_window = 3
# model_name_base = 'gpt-3.5-turbo'
model_name_base = 'gpt-3.5-turbo-0125'
model_name_16k = 'gpt-3.5-turbo-16k'
gpt_4_model = "gpt-4o"
# gpt_4_model = "gpt-4"
model_name = gpt_4_model
chunk_size = 1024
chunk_overlap = 128 
split_length_function = lambda x: len(tiktoken.encoding_for_model(model_name).encode(x))
separators=['。\n', '\n\n', '。', "\n", " ", ""]

# 文献检索
# embedding_model = "shibing624/text2vec-base-chinese"
# embedding_model = "moka-ai/m3e-base"
# embedding_model = "models/m3e-base"  # 768
embedding_model = "models/text2vec-base-chinese-paraphrase"  # 768
embedding = None#HuggingFaceEmbeddings(model_name=embedding_model)
# pinecone.init(
#     api_key='8c1c834c-db6e-45e2-b6dd-f5fdf74b378a',  # find at app.pinecone.io
#     environment='us-east1-gcp'  # next to api key in console
# )
index_name = "flowchart"

vertor_db_name = FAISS
retriever_name = MultiQueryRetriever

# Elasticsearch
# elasticsearch_index = "elasticsearch"
elasticsearch_index = "general_diagnosis_test"
# elasticsearch_url = "http://localhost:19201"
elasticsearch_url = "http://localhost:59201/"
es_user = "elastic"
es_password = "123456"

### Neo4j
neo4j_url = "neo4j://"
neo4j_username = "neo4j"
neo4j_password = "12345678"
