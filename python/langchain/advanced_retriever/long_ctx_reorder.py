from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

txts = [
    "치킨은 맛있다",
    "생선 찜은 맛이 없다",
    "스타크래프트는 좋은게임이다 ",
    "리니지는 나쁜게임이다"
]

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

reordering = LongContextReorder()

retriever = Chroma.from_texts(texts=txts, embedding=embedding_model).as_retriever(search_kwargs={"k": 10})

template = '''
----
{context}
----

"Answer in Korean no matter what."

Question: {query}
'''

document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)

prompt = PromptTemplate(
    template=template, input_variables=["context", "query"]
)


def print_by_docs(docs):
    for doc in docs:
        print(doc.page_content)


llm = ChatOllama(model="llama3:latest", temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt)

chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name="context"
)

while True:
    print("-" * 100)
    q = input()
    docs = retriever.get_relevant_documents(q)
    docs = reordering.transform_documents(docs)
    reordered_result = chain.run(input_documents=docs, query=q)
    result = chain.run(input_documents=docs, query=q)
    print(reordered_result)
