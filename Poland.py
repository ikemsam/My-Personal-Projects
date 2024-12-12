import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
if os.getenv("DEBUG_MODE", "false").lower() == "true":
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

# Constants
DOC_PATH = "./data/poland.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    ollama.pull(EMBEDDING_MODEL)  # Ensure embedding model is available
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Vector database loaded from disk.")
    else:
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("New vector database created.")
    return vector_db


def create_retriever_cached(vector_db):
    """Create a retriever with caching."""
    retriever = MultiQueryRetriever(vector_db)
    logging.info("Retriever created with caching.")
    return retriever


def create_chain(retriever, llm):
    """Create a chain with the retriever and language model."""
    chain = retriever | llm | StrOutputParser()
    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    st.title("Blulytix.ai Document Assistant")
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Processing..."):
            try:
                vector_db = load_vector_db()
                if not vector_db:
                    st.error("Failed to load or create the vector database.")
                    return

                retriever = create_retriever_cached(vector_db)
                llm = ChatOllama(model=MODEL_NAME)
                chain = create_chain(retriever, llm)
                response = chain.invoke(input=user_input)

                st.markdown("**Blulytix:**")
                st.write(response)
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()
