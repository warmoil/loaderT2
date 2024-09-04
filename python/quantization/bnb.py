import tiktoken
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOllama(model="llama3:latest")
tokenizer = tiktoken.get_encoding("cl100k_base")

template = '''
### [INST]
Instruction: Answer the question based on you knowledge . answer me in Korean no matter what. 
Here is context to help:

{context}

### QUESTION:
{question}
[/INST]
'''

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

loader = PyPDFLoader("../../data/pdf/nc.pdf")
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(pages)

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(texts, hf)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3}
)

rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
)

# result = rag_chain.invoke("혁신성장 정책 금융에서 인공지능이 중요한가?")
result = rag_chain.invoke("사람들은 치킨을 좋아하는가?")


for i in result['context']:
    print(f"주어진 근거: {i.page_content} / 출처: {i.metadata['source']} - {i.metadata['page']} \n\n")

print(f"\n답변: {result['text']}")