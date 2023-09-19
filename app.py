import streamlit as st
import openai
from dotenv import load_dotenv
from Chatbot import Chatbot
import os

st.title("Furnace Whisper")

# load the API key from the .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with st.sidebar:

   # pdfs = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)
    chatbot = Chatbot()
    
    docs = chatbot.load_directory("directory/")
    chunks = chatbot.split_text(docs)
    vectorstore = chatbot.vectorize(chunks)
    chain = chatbot.build_chain(vectorstore)
    
    st.write("Ready to chat!")
    st.write("Ask a question below and I'll try to answer it.")
    question = st.text_input("Question:")
    if question:
        chat_history = []

        response = chain({"question": question,  "chat_history": chat_history})
        chat_history.append((question, response["answer"]))
        st.write(response["answer"])
"""  
    if pdfs:
        text = chatbot.load_document(pdfs)
        chunks = chatbot.split_text(text)
        vectorstore = chatbot.vectorize(chunks)
        chain = chatbot.build_chain(vectorstore)
        st.write("Ready to chat!")
        st.write("Ask a question below and I'll try to answer it.")
        question = st.text_input("Question:")
        if question:
            chat_history = []

            response = chain({"question": question,  "chat_history": chat_history})
            chat_history.append((question, response["answer"]))
            st.write(response["answer"]) """