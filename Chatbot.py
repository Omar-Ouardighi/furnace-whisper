import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from dotenv import load_dotenv
from io import StringIO
from PyPDF2 import PdfReader
import os


class Chatbot:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=700 ,chunk_overlap=100)
        self.underlying_embeddings = OpenAIEmbeddings()
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


    def vectorize(self, directory):
        
        if os.path.exists("persist"):           
            vectorstore = Chroma(persist_directory="persist", embedding_function=self.underlying_embeddings)
            index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        else:
            loader = PyPDFDirectoryLoader(directory)
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}, text_splitter= self.splitter, ).from_loaders([loader])

        return index
    
    def build_chain(self, index):
        chain = ConversationalRetrievalChain.from_llm(llm = self.llm, retriever = index.vectorstore.as_retriever(search_kwargs={"k": 1}),
                                            return_source_documents=True)
        return chain
