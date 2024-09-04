import tiktoken
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(txt):
    tokens = tokenizer.encode(txt)
    return len(tokens)


def getLoader():
    return PyPDFLoader("../../../data/pdf/nc.pdf")


def getTextSplitter():
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=tiktoken_len)

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

def getHuggingFace():
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
