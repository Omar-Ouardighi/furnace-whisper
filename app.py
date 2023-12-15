import streamlit as st
import openai
from dotenv import load_dotenv
from Chatbot import Chatbot
import os
import utils
import shelve


st.set_page_config(page_title="ChitChatPDF", page_icon="📄")
st.header("ChitChatPDF")

utils.sidebar()
openai_api_key = st.session_state.get("OPENAI_API_KEY")

if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar. You can get a key at"
        " https://platform.openai.com/account/api-keys."
    )

if not utils.is_open_ai_key_valid(openai_api_key):
    st.stop()


pdf_docs = st.file_uploader(
            "Upload your pdf, txt  ", accept_multiple_files=True)

if pdf_docs:
    chatbot = Chatbot(openai_api_key, temperture=0, model_name="gpt-3.5-turbo", chunk_size=700 )
    text = chatbot.load_document(pdf_docs)
    vectorstore = chatbot.vectorize(text)
    chain = chatbot.build_chain(vectorstore, )
else:
    st.warning("Upload your file to get started.")

if not pdf_docs:
    st.stop()

if "conversation" not in st.session_state:
    st.session_state["conversation"] =  [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.chat_message(msg["role"]).write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

if "chat_history" not in st.session_state: 
    st.session_state["chat_history"] = []



if prompt := st.chat_input():
    
    st.session_state["conversation"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner("Processing"):
            
            response = chain({"question": prompt,  "chat_history": st.session_state["chat_history"]})     
            st.session_state.conversation.append({"role": "assistant", "content": response["answer"]})
            st.session_state["chat_history"].append((prompt, response["answer"]))
            st.chat_message("assistant", avatar="PW LOGO.png").write(response["answer"])
            

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages