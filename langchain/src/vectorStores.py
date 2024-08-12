import logging
from dataLoading import load_data, split_data
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
import os

# Set up logging
log_dir = "LOG"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(filename=os.path.join(log_dir, "/home/sat/RAG/langchain/log/vectorStore.log"), level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#membuat vector_store
def create_vectorstore(docs, persist_directory):
    logging.info('Creating vector store...')
    embedding_function = OllamaEmbeddings(model="llama3.1",show_progress=True, temperature=0)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    logging.info('Vector store created successfully.')
    return vectorstore

#loading vectorstore dari disk
def load_vectorstore(persist_directory):
    logging.info('Loading vector store...')
    embedding_function = OllamaEmbeddings(model="llama3.1")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    logging.info('Vector store loaded successfully.')
    return vectorstore

# persisitt_directory = "/home/sat/RAG/chroma"
# docs = load_data("/home/sat/RAG/data/UUD45 ASLI.pdf")
# docs   = split_data(docs)
# # qa = create_vectorstore(docs, persisitt_directory)
# qa = load_vectorstore(persisitt_directory)
# text = 'pasal 2'
# answer = qa.similarity_search(text)
# print(answer)
