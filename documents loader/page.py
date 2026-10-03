from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content.encode("utf-8", errors="replace").decode("utf-8"))