import streamlit as st
import openai
from dotenv import load_dotenv
from Chatbot import Chatbot
import os



st.title("Furnace Whisper")

# load the API key from the .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chatbot = Chatbot()


vectorstore = chatbot.vectorize("directory/")
chain = chatbot.build_chain(vectorstore)

if "conversation" not in st.session_state:
    st.session_state["conversation"] =  [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.conversation:
    st.chat_message(msg["role"]).write(msg["content"])

if "chat_history" not in st.session_state: 
    st.session_state["chat_history"] = None

if prompt := st.chat_input():
    st.session_state["conversation"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner("Processing"):
        chat_history = []
        response = chain({"question": prompt,  "chat_history": chat_history})
        st.session_state.conversation.append({"role": "assistant", "content": response["answer"]})
        st.chat_message("assistant").write(response["answer"])   

    


