import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import json

load_dotenv()

st.sidebar.title("🌍 Global Guru")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# File to store API keys
API_KEYS_FILE = "api_keys.json"

# Function to load API keys from file
def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as file:
            return json.load(file)
    return {"groqApi": "", "geminiApi": ""}

# Function to save API keys to file
def save_api_keys(groqApi, geminiApi):
    with open(API_KEYS_FILE, "w") as file:
        json.dump({"groqApi": groqApi, "geminiApi": geminiApi}, file)

# Load existing API keys
api_keys = load_api_keys()

# Check if API keys are already provided
groqApi_provided = bool(api_keys.get("groqApi"))
geminiApi_provided = bool(api_keys.get("geminiApi"))

if groqApi_provided and geminiApi_provided:
    st.sidebar.success("✅ API keys already provided.")
else:
    groqApi = st.sidebar.text_input("Enter Your Groq Api", type="password")
    geminiApi = st.sidebar.text_input("Enter Your Gemini Pro Api", type="password")
    apiSubmit = st.sidebar.button("Submit Api Keys")

    if apiSubmit:
        save_api_keys(groqApi, geminiApi)
        os.environ["GOOGLE_API_KEY"] = geminiApi
        st.sidebar.success("✅ Groq Api Key Submitted")
        st.sidebar.success("✅ Gemini Api Key Submitted")

# Initialize ChatGroq if API keys are provided
if groqApi_provided and geminiApi_provided:
    groq_api_key = api_keys.get("groqApi")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="Llama3-8b-8192"
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context> 
        Questions:{input}
        """
    )

    def vector_embedding():
        if "vectors" not in st.session_state:
            with st.spinner("Creating document embeddings..."):
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
                st.session_state.loader = PyPDFDirectoryLoader("./docs")  # Data Ingestion
                st.session_state.docs = st.session_state.loader.load()  # Document Loading
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

    # Button to create document embeddings
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    if st.sidebar.button("📚 Create Document Embeddings"):
        vector_embedding()
        st.sidebar.success("✅ Vector Store DB is ready.")
        st.sidebar.success("✅ Now You Can Ask The Question.")
        st.sidebar.success("✅ Good To Go.")

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # Custom CSS for centering the text
    st.markdown("""
        <style>
        .center-text {
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar content with centered text
    st.sidebar.markdown("<h2 class='center-text'>Developed with ❤️ for GenAI by <a href='https://www.linkedin.com/in/aryan-tiwari-174a50298/' style='text-decoration:none;'>Aryan Tiwari</a></h2>", unsafe_allow_html=True)

    # Input from user
    user_prompt = st.chat_input("Ask To Global Guru")

    # Display the question and get the answer if the button is clicked
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        try:
            response = retrieval_chain.invoke({'input': user_prompt})
            st.chat_message("bot").markdown(response['answer'])
            st.session_state.messages.append({"role": "bot", "content": response['answer']})
        except Exception as e:
            st.error(f"Error retrieving answer: {str(e)}")
else:
    st.sidebar.warning("Please provide API keys to access the chatbot functionality.")
