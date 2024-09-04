from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

loaders = [
    PyPDFLoader("../../../data/pdf/nc.pdf"),
    PyPDFLoader("../../../data/pdf/nc.pdf")
]

docs = []

for loader in loaders:
    docs.extend(loader.load_and_split())

txt_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)

txts = txt_splitter.split_documents(docs)

bm25_retriever = BM25Retriever.from_documents(txts)
bm25_retriever.k = 2

faiss_vectorstore = FAISS.from_documents(txts, embedding_model)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

llm = ChatOllama(model="llama3:latest", temperature=0)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    # retriever=ensemble_retriever,
    retriever=faiss_retriever,
    return_source_documents=True
)

template = '''
"Answer in Korean no matter what."

Question: {question}
'''
def print_metadata_by_documents(docs):
    for doc in docs:
        print(doc.metadata)


while True:
    print("-" * 100)
    q = input()
    result = qa(template.format(question=q))
    print(result["result"])
    print_metadata_by_documents(result["source_documents"])
# for doc in docs:
#     print(doc.metadata)
#     print(":")
#     print(doc.page_content)
#     print("-" * 100)
#
# print("총:" + str(len(docs)) + "건")
