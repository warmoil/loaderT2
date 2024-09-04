from datetime import datetime, timedelta

import faiss
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)
embedding_size = 768

index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embedding_model, index, InMemoryDocstore({}), {})

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.4, k=1
)

yesterday = datetime.now() - timedelta(days=1)

retriever.add_documents(
    [Document(page_content="한국어는 훌륭합니다", metadata={"last_accessed_at": yesterday})]
)

retriever.add_documents(
    [Document(page_content="영어는 훌륭합니다", metadata={"last_accessed_at": datetime.now()})]
)

result = retriever.get_relevant_documents("한국어가 좋아요")
print(result[0].page_content)
