import sys
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()  # Load environment variables from .env file


data = TextLoader("documents loader/notes.txt")
docs=data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)


template = ChatPromptTemplate([
    ('system', "You are a helpful assistant."),
    ('human', "{data}"),
])



model=ChatMistralAI(model="ministral-8b-latest") 

prompt=template.format_messages(data=docs[0].page_content)

result=model.invoke(prompt)  # Example invocation of the model

print(result.content)  