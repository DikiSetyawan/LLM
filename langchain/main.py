from src.dataLoading import load_data, split_data
from src.vectorStores import create_vectorstore, load_vectorstore
from src.retrieval import retriever


def main(question):
    raw_documents = load_data('/home/sat/RAG/data/UUD45 ASLI.pdf')
    persist_directory = "/home/sat/RAG/chroma"
    processed_docs = split_data(raw_documents)
    vectorstore = create_vectorstore(processed_docs, persist_directory=persist_directory)
    chains = retriever(vectorstore)
    result  = chains.invoke(question)
    return result

questions = "jelasakan tentang pasal 31 pada UUD"
res = main (questions)
print (res)
