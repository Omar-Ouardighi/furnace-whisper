import streamlit as st
from dotenv import load_dotenv
import openai
import os
import shelve
load_dotenv()

def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n" 
            "2. Upload your file 📄\n" 
            "3. Ask a question 💬\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )


        st.session_state["OPENAI_API_KEY"] = api_key_input

     

def is_open_ai_key_valid(openai_api_key) -> bool:

    if not openai_api_key:
        return False
    try:
        
      
        chat_completion = openai.ChatCompletion.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say this is a test",
                }
            ],
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        openai.api_key = os.getenv('OPENAI_API_KEY')

    except Exception as e:
        st.error(f"{e.__class__.__name__}: {e}")
        return False

    return True


# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("conversation", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["conversation"] = messages