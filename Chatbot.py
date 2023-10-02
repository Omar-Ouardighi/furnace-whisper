import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts.prompt import PromptTemplate

from dotenv import load_dotenv
from io import StringIO
from PyPDF2 import PdfReader
import os


class Chatbot:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=700 ,chunk_overlap=100)
        self.underlying_embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")

        qa_template = """
        You are "Furnace Whisper," an AI expert on blast furnaces. When presented with a file or context by the user, utilize it to guide your responses. 
        Ensure accuracy by only answering based on the context given; if unsure, respond with "I don't know." 
        If a question is outside the context or unrelated to blast furnaces, politely inform the user that you are tailored to address topics within the provided context and blast furnace realm.
        Always prioritize detailed answers while maintaining a courteous tone.

        context: {context}
        =========
        question: {question}
        ======
        """

        self.QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])   

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
        
        if os.path.exists("./persist"):           
            vectorstore = Chroma(persist_directory="./persist", embedding_function=self.underlying_embeddings)
            
        else:
            loader = PyPDFDirectoryLoader(directory)
            documents = loader.load()
            chunks = self.splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(chunks, self.underlying_embeddings, persist_directory="./persist")

        return vectorstore
    
    def build_chain(self, vectorstore):
        chain = ConversationalRetrievalChain.from_llm(llm = self.llm, retriever = vectorstore.as_retriever(search_kwargs={"k": 3}),
                                    combine_docs_chain_kwargs={'prompt': self.QA_PROMPT},
                                            return_source_documents=True)
        return chain
