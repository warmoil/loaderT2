import time

import tiktoken
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(txt):
    tokens = tokenizer.encode(txt)
    return len(tokens)


loader = PyPDFLoader("../../../data/pdf/nc.pdf")
pages = loader.load_and_split()

txt_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=tiktoken_len)
txts = txt_splitter.split_documents(pages)

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

template = '''
"You are a helpful, professional assistant named Bro. Introduce yourself first, and answer the questions. answer me in Korean no matter what. "


Question: {question}
Helpful Answer:"""

'''

prompt = PromptTemplate.from_template(template=template)

llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)


chain = RetrievalQA.from_chain_type(llm=llm_chain, chain_type="stuff", retriever=docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10},
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt
    }
), return_source_documents=True)

while True:
    print("-" * 100)
    print("질문을 입력 하세요.", end="")
    q = input()
    start_at = time.time()
    try:
        result = chain(q)
        print("\t\t 걸린시간" + str(time.time() - start_at))
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        # Add debug print to check the response
        response = llm_chain.predict({"question": q})
        print("LLM response:", response)
