import streamlit as st 
from langchain.document_loaders import PyPDFDirectoryLoader
import os
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from streamlit import session_state
from dotenv import load_dotenv
load_dotenv() 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


prompt_template = """
  Answer the question as detailed as possible from the provided context, make sure to provide all the details from the given context only, 
  Break your answer up into nicely readable paragraphs. \n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
""" 

st.set_page_config(page_title="My Streamlit App",initial_sidebar_state="expanded",)
st.header("chat with pdfs🥳")


def main():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    uploaded_pdf = st.file_uploader("Upload an pdf", type=["pdf"])
    # ****************************************************************
    def embed():

        # --------------------------Saving the uploaded file--------------------------------------
        def save(uploaded_file):
            pdfs_path = "D:\\vs_code_projects\\PDF_Langchain\\pdfs"
            Check if the file already exists.
            Get a list of all files in the directory
            file_list = os.listdir(pdfs_path)
            Iterate through the files and delete them
            for file_name in file_list:
                file_path = os.path.join(pdfs_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            if uploaded_file is not None:
                # Check if "pdfs" exists and handle accordingly
                if not os.path.exists(pdfs_path):
                    os.makedirs(pdfs_path)
                    
                with open(os.path.join(pdfs_path, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

        save(uploaded_pdf)
        # ------------------------------File Saving Done ----------------------------------------------

        # ------------------------ Loading/ Splitting in shunks/ Generate Emebeddings----------------- 
        loader = PyPDFDirectoryLoader("D:\\vs_code_projects\\PDF_Langchain\\pdfs")
        loader = PdfReader(uploaded_pdf)
        data = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        context = "\n".join(str(p.page_content) for p in data)

        texts = text_splitter.split_text(data)
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

        # -------------------------------------------------------------------------------------------

        return vector_index


    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # *******************************************************************
    
    def queries(question):
        vector_index = embed()
        docs = vector_index.get_relevant_documents(question)
        # print(docs)
        response = chain(
            {"input_documents":docs, "question": question},
            return_only_outputs=True)
        
        return response

    # ****************************************



    question=st.text_input("Your Question ",key="input")
    submit=st.button("Ask Question... 🤔")

    if submit :
        # st.image(uploaded_pdf)
        response  = ""
        print("working...", question[:50])
        response = queries(question)
        st.subheader("Response:")
        st.write(response['output_text'])
        # Inserting the chat response for history
        st.session_state['chat_history'].insert(0,("Bot", response['output_text']))
        # Inserting the question for chat history
        st.session_state['chat_history'].insert(0,("You", question))
    
    st.subheader("Chat History:")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")
   
if __name__ == "__main__":
    main()
