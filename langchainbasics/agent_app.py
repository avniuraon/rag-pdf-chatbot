import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# PDF imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ---------------- UI ----------------
st.set_page_config(page_title="Agentic AI")
st.title("🧠 Multi-Tool AI Agent")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ---------------- PDF PROCESSING ----------------
retriever = None

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever()

# ---------------- TOOLS ----------------

@tool
def calculator(query: str) -> str:
    """Useful for math calculations"""
    try:
        return str(eval(query))
    except:
        return "Invalid math expression"

@tool
def pdf_qa(query: str) -> str:
    """Answer questions from uploaded PDF"""
    if retriever:
        docs = retriever.invoke(query)
        return "\n".join([doc.page_content for doc in docs])
    return "No PDF uploaded"

tools = [calculator, pdf_qa]

# ---------------- AGENT ----------------

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---------------- CHAT ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = agent.run(prompt)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})