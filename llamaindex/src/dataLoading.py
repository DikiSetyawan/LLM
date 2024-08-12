from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import logging
import torch

# Configure logging
logging.basicConfig(
    filename='process.log',  # Log file
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def convert_to_txt(filepath):
    """Convert a file to .txt format and delete the original non-txt file."""
    base, ext = os.path.splitext(filepath)
    if ext.lower() != '.txt':
        txt_filepath = base + '.txt'
        try:
            with open(filepath, 'r') as original_file:
                data = original_file.read()
            with open(txt_filepath, 'w') as txt_file:
                txt_file.write(data)
            os.remove(filepath)
            logging.info(f"Converted {filepath} to {txt_filepath} and deleted the original file.")
        except Exception as e:
            logging.error(f"Failed to convert {filepath} to txt: {e}")

def load_documents(filepath):
    # Ensure the directory exists
    if not os.path.exists(filepath):
        logging.error(f"The directory {filepath} does not exist.")
        return []
    
    # Process each file in the directory
    for filename in os.listdir(filepath):
        full_path = os.path.join(filepath, filename)
        if os.path.isfile(full_path):
            convert_to_txt(full_path)
    
    logging.info(f"Loading documents from: {filepath}")
    documents = SimpleDirectoryReader(filepath).load_data()
    logging.info(f"Loaded {len(documents)} documents.")
    return documents
    
def preprocess(documents, batch_size=100):
    logging.info(f"Starting preprocessing on {len(documents)} documents with batch size: {batch_size}")
    text_splitter = TokenTextSplitter(chunk_size=1024)
    embed_model = HuggingFaceEmbedding(model_name="/home/sat/RAG/llamaindex/model/all-mpnet-base-v2", device="cuda")
    
    logging.info(f"Model loaded and moved to CUDA")

    processed_documents = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1} with {len(batch)} documents")
        
        text_chunks = [text_splitter.split_text(doc) for doc in batch]
        flattened_chunks = [item for sublist in text_chunks for item in sublist]
        logging.info(f"Split into {len(flattened_chunks)} text chunks")

        embeddings = embed_model.embed_documents(flattened_chunks)
        logging.info(f"Generated embeddings for {len(flattened_chunks)} text chunks")
        
        # Create nodes and append to processed_documents
        # Example: processed_documents.extend(create_nodes(embeddings))

    logging.info("Preprocessing complete.")
    return processed_documents