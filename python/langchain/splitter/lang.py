from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language
)

kt_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.KOTLIN,
    chunk_size=200,
    chunk_overlap=0,
)
with open("../../../data/code/ChatService.kt") as f:
    kt_code = f.read()

docs = kt_splitter.create_documents([kt_code])


for doc in docs:
    print(doc)
    print("-"*100)
