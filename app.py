import streamlit as st
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter




st.title("Furnace Whisper")

st.sidebar.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)
