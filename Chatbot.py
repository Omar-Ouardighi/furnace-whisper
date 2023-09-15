import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from lanngchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv


class Chatbot:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000 ,chunk_overlap=100)
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    def load_document(self, files):
        text=""
        for file in files:
            pdf = PyMuPDFLoader(file)
            for page in pdf:
                text += page.page_content
        return text

    
    def split_text(self, text):
        return self.splitter.split_text(text)


    def vectorize(self, chunks):
        vectorestore = Chroma.from_text(chunks, self.embeddings)
        return vectorestore
    
    def build_chain(self, vectorstore):
        chain = ConversationalRetrievalChain(llm = self.llm, retriever = vectorstore.as_retriever(),
                                            return_source_documents=True)
        return chain

    