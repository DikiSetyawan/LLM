from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import os
import tiktoken

# Create a directory for the log files
log_dir = "LOG"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging to a file
logging.basicConfig(filename=os.path.join(log_dir, "/home/sat/RAG/langchain/log/loadingData.log"), level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(path):
    logging.info(f"Loading data from {path}...")
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    logging.info(f"Loaded {len(pages)} pages from {path}")
    return pages

def split_data(pages):
    logging.info("Splitting data into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1024,
        chunk_overlap=0,
        length_function=len
    )
    docs = splitter.split_documents(pages)
    logging.info(f"Split data into {len(docs)} chunks")
    return docs



# raw_documents = load_data('/home/sat/RAG/data/UUD45 ASLI.pdf')
# processed_docs = split_data(raw_documents)
# for doc in processed_docs:
#     print(doc.page_content)
