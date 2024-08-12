from src.dataLoading import load_documents, preprocess
from src.dataingestion import convert_pdf_to_txt



documents = load_documents("/home/sat/RAG/llamaindex/data")
processed_documents = preprocess(documents)

print(len(processed_documents))