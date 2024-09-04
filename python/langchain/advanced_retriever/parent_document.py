from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loaders = [
    PyPDFLoader("../../../data/pdf/nc.pdf"),
    PyPDFLoader("../../../data/pdf/nc.pdf")
]
docs = []

for loader in loaders:
    docs.extend(loader.load_and_split())

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
model_kwargs = {"device": "cpu"}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
vector_store = Chroma(
    # collection_name="full_documents", embedding_function=hf
    collection_name="split_parents", embedding_function=hf
)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    docstore=store,
    vectorstore=vector_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
retriever.add_documents(docs, ids=None)

sub_docs = vector_store.similarity_search("군대")

print(len(list(store.yield_keys())))

print("글길이:{}\n".format(len(sub_docs[0].page_content)))
print(sub_docs[1].page_content)

retrieved_docs = retriever.get_relevant_documents("군대")
print("글길이:{}\n".format(len(retrieved_docs[0].page_content)))
print(retrieved_docs[1].page_content)
