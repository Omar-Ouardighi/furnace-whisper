import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFDirectoryLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from io import StringIO
from PyPDF2 import PdfReader


class Chatbot:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000 ,chunk_overlap=100)
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    def load_document(self, files):
        text=""
        for file in files:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text()
        return text
    
    def load_directory(self, directory):
        loader = PyPDFDirectoryLoader(directory)
        return loader.load()

    
    def split_text(self, text):
        return self.splitter.split_documents(text)


    def vectorize(self, chunks):
        vectorestore = Chroma.from_documents(chunks, self.embeddings)
        return vectorestore
    
    def build_chain(self, vectorstore):
        chain = ConversationalRetrievalChain.from_llm(llm = self.llm, retriever = vectorstore.as_retriever(),
                                            return_source_documents=True)
        return chain

    