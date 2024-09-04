from langchain.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


embedding_model = OllamaEmbeddings(model="llama3")

with open("../../../data/txt/nc.txt") as f:
    stateOfTheUnion = f.read()

txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len
)

txts = txt_splitter.split_text(stateOfTheUnion)

embeddings = embedding_model.embed_documents(
    txts
)

q = embedding_model.embed_query("리니지의 대한 평가는?")


print(len(embeddings),len(embeddings[0]))
print(embeddings)
