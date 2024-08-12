from vectorStores import create_vectorstore, load_vectorstore
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable import RunnablePassthrough
import os 
import logging

log_dir = "LOG"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging to a file
logging.basicConfig(filename=os.path.join(log_dir, "/home/sat/RAG/langchain/log/retreival.log"), level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

template = """
    You are a knowledgeable expert in. 
    Access and process information from the provided context, ensuring confidentiality. 
    Deliver concise, accurate, and actionable responses to the query. 
    Avoid making assumptions or generating speculative content.
    Quuestion : {question}
    Context : {context}
    """
prompt = ChatPromptTemplate.from_template(template)




def retriever(vectorstore):
    logging.info('Set LLM')
    llm = ChatOllama(model= 'llama3.1', temperature = 0.1)
    logging.info('LLM setup successfully.using ollama (llama3.1)') 
    logging.info('---------------------------')
    logging.info('Retriever setup...')
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                         search_kwargs={"score_threshold": 0.7})
    logging.info('Retriever setup successfully.')
    logging.info('---------------------------')
    chains = {"context": retriever, "question": RunnablePassthrough()}| prompt | llm | StrOutputParser()
    return chains


