import os
import time

from langchain.document_loaders import PyPDFLoader

startTime = time.time()
current_path = os.getcwd()
print("경로:"+current_path)
loader = PyPDFLoader(current_path+"/data/pdf/nc.pdf")

pages = loader.load_and_split()
file = open(current_path+"/data/txt/nc.txt", "w")
for page in pages:
    file.write(page.page_content)
total = time.time() - startTime
print("총 걸린 시간:" + str(total))
