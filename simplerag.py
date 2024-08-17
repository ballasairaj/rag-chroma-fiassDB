## Data Ingestion
from langchain_community.document_loaders import TextLoader
loader = TextLoader("speech.txt")
text_documents = loader.load()
print(text_documents)

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=("post-title", "post-content", "post-header")
                       )))

text_documents = loader.load()
print(text_documents)

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('fswd.pdf')  # Load the PDF file in same directory
docs = loader.load()
print(docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
print(documents[:5])

# Chroma vector database

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set the environment variable for OpenAI API key
os.environ['OPENAI_API_KEY'] = 'place_your_api_here'  # place your api key here
print("Environment variable set successfully.")

# Ensure 'documents' is defined
if 'documents' not in locals():
    raise ValueError("The variable 'documents' is not defined.")
print("Documents variable is defined.")

# Reduce the size of the dataset to avoid memory issues
documents_subset = documents[:5]
print(f"Using a subset of documents: {len(documents_subset)} documents.")

# Create the Chroma instance from documents and embeddings
try:
    embeddings = OpenAIEmbeddings()
    print("OpenAIEmbeddings instance created successfully.")
    
    # Debugging: Print the first document to check its structure
    print(f"First document: {documents_subset[0]}")
    
    db = Chroma.from_documents(documents_subset, embeddings)
    print("Chroma instance created successfully.")
except MemoryError as me:
    print(f"MemoryError: {me}")
except ImportError as ie:
    print(f"ImportError: {ie}")
except Exception as e:
    print(f"An error occurred: {e}")

# Ensure 'db' is defined before running the query
if 'db' in locals():
    query = "What is the definition of a set?"
    try:
        retrieved_results = db.similarity_search(query)
        print(retrieved_results[0].page_content)
    except Exception as e:
        print(f"An error occurred during the query: {e}")
else:
    print("The variable 'db' is not defined.")

# FAISS Vector Database
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
db1 = FAISS.from_documents(documents[:10], OpenAIEmbeddings())

query = "what is firefox"
results = db1.similarity_search(query)
print(results[0].page_content)