import streamlit as st
from src.dataLoading import load_data, split_data
from src.vectorStores import create_vectorstore, load_vectorstore, add_data_to_vectorstore
from src.retrieval import retriever
import os

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def main(question):
    persist_directory = "/home/sat/RAG/chroma"
    
    # Check if the vectorstore already exists
    if os.path.exists(persist_directory):
        st.write("Vectorstore already exists. Loading existing vectorstore.")
        vectorstore = load_vectorstore(persist_directory)
    else:
        st.write("Vectorstore not found. Creating new vectorstore.")
        raw_documents = load_data('/home/sat/RAG/data/UUD45 ASLI.pdf')
        processed_docs = split_data(raw_documents)
        vectorstore = create_vectorstore(processed_docs, persist_directory=persist_directory)
    
    chains = retriever(vectorstore)
    result = chains.invoke(question)
    return result

st.title("Chat with AI")

# Input field for the question
question = st.text_input("You:")

if st.button("Send"):
    if question:
        # Get the answer
        answer = main(question)
        payload = {"question": question, "answer": answer}
        
        # Append the question and answer to the chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Clear the input field after submission
        st.text_input("You:", value="", key="clear")

# Display the chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Assistant:** {chat['content']}")

# Button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.write("Chat history cleared!")