from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://n.news.naver.com/article/014/0005218435?cds=news_media_pc&type=editn")

data = loader.load()

print(data[0].page_content.replace("\n\n","\n"))