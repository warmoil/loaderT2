import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(txt):
    tokens = tokenizer.encode(txt)
    return len(tokens)

txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=tiktoken_len
)


with open("../../../data/txt/nc.txt") as f:
    stateOfTheUnion = f.read()

txts = txt_splitter.split_text(stateOfTheUnion)

char_list = []

for txt in txts:
    print(txt)
    print("*"*100)
    char_list.append(tiktoken_len(txt))

print(char_list )