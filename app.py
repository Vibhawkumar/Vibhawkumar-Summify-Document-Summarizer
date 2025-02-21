import streamlit as st 
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
persist_directory = "./chroma_db"  # Create a folder for persistence
chroma_client = Chroma(persist_directory=persist_directory)
from chromadb import Client
chroma_client = Client(tenant="default_tenant", database="default_database")
from langchain_community.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv() 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Chat with PDFs 🥳", initial_sidebar_state="expanded")
st.header("Chat with PDFs 🥳")

# Define the prompt template
prompt_template = """
  Answer the question as detailed as possible from the provided context.
  Ensure to provide all the details from the given context only.
  Break your answer into readable paragraphs.\n\n
  Context:\n {context}?\n
  Question: \n{question}\n
  Answer:
"""

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def save_uploaded_file(uploaded_file):
    """Save uploaded file and delete previous PDFs."""
    pdfs_path = "./pdfs"

    # Delete existing files
    if os.path.exists(pdfs_path):
        for file_name in os.listdir(pdfs_path):
            file_path = os.path.join(pdfs_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Save the new uploaded file
    if uploaded_file is not None:
        if not os.path.exists(pdfs_path):
            os.makedirs(pdfs_path)

        file_path = os.path.join(pdfs_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    return None

def embed(uploaded_file):
    """Process and embed the PDF content."""
    file_path = save_uploaded_file(uploaded_file)
    
    if not file_path:
        return None

    # Load the document
    loader = PyPDFDirectoryLoader("./pdfs")
    documents = loader.load()

    # Extract text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_documents(texts, embeddings).as_retriever()

    return vector_index

def queries(question, vector_index):
    """Retrieve relevant documents and generate a response."""
    if vector_index is None:
        return "No document found. Please upload a PDF first."

    docs = vector_index.get_relevant_documents(question)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response.get("output_text", "No response generated.")

def main():
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        vector_index = embed(uploaded_pdf)
        st.session_state["vector_index"] = vector_index  # Store for later queries

    question = st.text_input("Your Question", key="input")
    submit = st.button("Ask Question 🤔")

    if submit:
        if "vector_index" in st.session_state:
            response = queries(question, st.session_state["vector_index"])
        else:
            response = "Please upload a PDF first."

        st.subheader("Response:")
        st.write(response)

        # Store chat history
        st.session_state["chat_history"].insert(0, ("Bot", response))
        st.session_state["chat_history"].insert(0, ("You", question))

    # Display chat history
    st.subheader("Chat History:")
    for role, text in st.session_state["chat_history"]:
        st.write(f"{role}: {text}")
        print(text)


if __name__ == "__main__":
    main()
