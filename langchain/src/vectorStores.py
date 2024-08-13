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


from src.dataLoading import load_data, split_data
from src.vectorStores import load_vectorstore, update_vectorstore
import os

def add_data_to_vectorstore(new_data_path, persist_directory):
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vectorstore not found at {persist_directory}. Please create a vectorstore first.")

    # Load the existing vectorstore
    vectorstore = load_vectorstore(persist_directory)

    # Load and process the new data
    new_documents = load_data(new_data_path)
    processed_new_docs = split_data(new_documents)

    # Update the vectorstore with the new data
    updated_vectorstore = update_vectorstore(vectorstore, processed_new_docs, persist_directory)

    return updated_vectorstore

# persisitt_directory = "/home/sat/RAG/chroma"
# docs = load_data("/home/sat/RAG/data/UUD45 ASLI.pdf")
# docs   = split_data(docs)
# # qa = create_vectorstore(docs, persisitt_directory)
# qa = load_vectorstore(persisitt_directory)
# text = 'pasal 2'
# answer = qa.similarity_search(text)
# print(answer)
