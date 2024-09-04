from langchain.text_splitter import CharacterTextSplitter

with open("../../../data/txt/nc.txt") as f:
    stateOfTheUnion = f.read()

txtSplitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=900,
    chunk_overlap=100,
    length_function=len
)

txts = txtSplitter.split_text(stateOfTheUnion)

char_list = []

for txt in txts:
    print(txt)
    print("-"*100)
    char_list.append(len(txt))
print(len(txts))
print(char_list)