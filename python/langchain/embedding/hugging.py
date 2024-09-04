import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy import dot
from numpy.linalg import norm


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


start = time.time()

with open("../../../data/txt/nc.txt") as f:
    stateOfTheUnion = f.read()

txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

# txts = txt_splitter.split_text(stateOfTheUnion)

txts = [
    "나는 집에 가고 싶어요",
    "나는 치킨을 좋아합니다",
    "나의 이름은 스타게이저 입니다"
]

embedding_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

embeddings = embedding_model.embed_documents(
    txts
)

embedding_finish_time = time.time()

print("embedding 완료 걸린시간:" + str(embedding_finish_time - start))

q_str = "내가 좋아하는 음식은?"
q = embedding_model.embed_query(q_str)

i = 0

print("질문:\"{}\"의 유사도 \n".format(q_str),"-"*100)
for embedding in embeddings:

    score = round(cos_sim(q, embedding), 3)

    print("문장:\"{}\"의 유사도: {}".format(txts[i],str(score)))

# if (score > 0.5):
    #     print("유사도" + str(cos_sim(embedding, q)))
    #     print(txts[i])
    #     print("-" * 100)
    i = i + 1

print("embedding 후 걸린시간"+str(time.time()-embedding_finish_time))