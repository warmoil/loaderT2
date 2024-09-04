from langchain.document_loaders import UnstructuredURLLoader

urls = [
    "https://n.news.naver.com/article/014/0005218435?cds=news_media_pc&type=editn",
    "https://n.news.naver.com/article/003/0012687323?cds=news_media_pc&type=editn"
]

loader = UnstructuredURLLoader(urls=urls)


data = loader.load()

print(data)