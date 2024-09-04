from langchain_text_splitters import RecursiveCharacterTextSplitter

txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len
)

with open("../../../data/txt/nc.txt") as f:
    stateOfTheUnion = f.read()

txts = txt_splitter.split_text(stateOfTheUnion)

char_list = []

for txt in txts:
    print(txt)
    print("-"*100)
    char_list.append(len(txt))
print(len(txts))
print(char_list)