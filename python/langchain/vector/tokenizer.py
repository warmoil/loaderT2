import os
import time

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(txt):
    tokens = tokenizer.encode(txt)
    return len(tokens)


start_at = time.time()

# load the docs and split into chunks
loader = PyPDFLoader("../../../data/pdf/nc.pdf")
pages = loader.load_and_split()

# split it into chinks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=tiktoken_len)
docs = text_splitter.split_documents(pages)

load_at = time.time()
print("스플릿 까지 걸린시간: " + str(load_at - start_at))  # create the open-source embedding function

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

hugging_load_at = time.time()
print("huggingFace 준비시간: " + str(hugging_load_at - load_at))

# load it into chroma

# db = Chroma.from_documents(docs, hf)

# query and result
query = "갈등의 뿌리는?"
# docs = db.similarity_search(query)
#
# print(docs[0].page_content)
# print(len(docs))

# save to disk
# db2 = Chroma.from_documents(docs, hf,persist_directory="../../data/vector/chroma_db/aphg")
# docs = db2.similarity_search(query)
# print(docs[0])


query = "nc는 향후 어떻게 될까?"
# load from disk
db3 = Chroma(persist_directory="../../data/vector/chroma_db/aphg", embedding_function=hf)
# docs = db3.similarity_search(query)
docs = db3.similarity_search_with_score(query, k=3)
print(docs[0][1])
print("-" * 100)
# docs = db3.similarity_search("tv에 나오는 채널은?")
docs = db3.similarity_search_with_score("리니지 평가는?",k=3)
print(docs[0][1])

print("hugging 준비후 걸린시간: " + str(time.time() - hugging_load_at))
