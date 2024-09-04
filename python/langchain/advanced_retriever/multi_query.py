import logging

from langchain.retrievers import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://m.blog.naver.com/tooc/222547939279")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# VectorDB
model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
model_kwargs = {"device": "cpu"}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb = Chroma.from_documents(documents=splits, embedding=ko_embedding)

question = "워치 울트라는 어때?"
llm = ChatOllama(temperature=0, model="llama3:latest")
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

while True:
    print("-" * 100 + "\n질문: ", end="")
    q = input()
    unique_docs = retriever_from_llm.get_relevant_documents(query=q)
    print(len(unique_docs))
    print(unique_docs)