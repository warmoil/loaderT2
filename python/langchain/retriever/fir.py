import time

import tiktoken
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(txt):
    tokens = tokenizer.encode(txt)
    return len(tokens)


loader = PyPDFLoader("../../../data/pdf/nc.pdf")
pages = loader.load_and_split()

txt_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=tiktoken_len)
txts = txt_splitter.split_documents(pages)
# txt_splitter.split_text()
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

docsearch = Chroma.from_documents(txts, hf)

llm = ChatOllama(model="llama3:latest")

chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10},
    verbose=True,
), return_source_documents=True,)

template = '''
"Answer in Korean no matter what."

Question: {question}
helpful Answer:
'''

while True:
    print("-" * 100)
    print("질문을 입력 하세요.", end="")
    q = input()
    start_at = time.time()
    # result = chain(q)
    query = template.format(question=q)
    result = chain(query)
    print("\t\t 걸린시간" + str(time.time() - start_at))
    print(result['result'])
    print(result)
