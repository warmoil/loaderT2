from langchain.vectorstores import FAISS

from python.langchain.vector.baseUtils import getHuggingFace

# load
# loader = getLoader()
# pages = loader.load_and_split()
#
# # split
# text_splitter = getTextSplitter()
# docs = text_splitter.split_documents(pages)

# hugging face
hf = getHuggingFace()

# db = FAISS.from_documents(documents=docs, embedding=hf)
#
# docs = db.similarity_search_with_score("최근 업데이트?", k=3)
# print(docs[0][0])
# print("유사도" + str(docs[0][1]))
#
# db.save_local("../../data/vector/faiss/aphg")

newDb = FAISS.load_local("../../../data/vector/faiss/aphg", hf, allow_dangerous_deserialization=True)

docs = newDb.similarity_search_with_score("공선전이 뭐야??", k=3)
print(docs[0][0])
print("유사도" + str(docs[0][1]))

docs = newDb.max_marginal_relevance_search("최근 업데이트 내용은?", k=3)
print(docs[0].page_content)
print("출처:" + str(docs[0].metadata))