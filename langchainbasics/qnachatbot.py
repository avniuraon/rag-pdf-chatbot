import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# PDF imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page config
st.set_page_config(page_title="Groq Chatbot")
st.title("CHATBOT ")

# Sidebar
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input("GROQ API Key", type="password")

    model_name = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "gemma2-9b-it"
        ],
        index=0
    )

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------- PDF PROCESSING --------
@st.cache_resource
def process_pdf(uploaded_file):
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

    return vectorstore

retriever = None
if uploaded_file:
    vectorstore = process_pdf(uploaded_file)
    retriever = vectorstore.as_retriever()

# -------- MODEL --------
def get_llm(api_key, model_name):
    try:
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            streaming=True
        )
    except Exception:
        return ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.7,
            streaming=True
        )

def get_chain(api_key, model_name):
    if not api_key:
        return None

    llm = get_llm(api_key, model_name)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use context if available."),
        ("placeholder", "{chat_history}"),
        ("user", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    return prompt | llm | StrOutputParser()

chain = get_chain(api_key, model_name)

# -------- UI --------
if not chain:
    st.warning("Enter your Groq API key")
else:
    # show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # input
    if question := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        # build chat history
        chat_history = [
            ("user" if m["role"] == "user" else "assistant", m["content"])
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # context from PDF
            context = ""
            if retriever:
                docs = retriever.invoke(question)
                context = "\n".join([doc.page_content for doc in docs])

                
                sources = []
                for doc in docs:
                    if "page" in doc.metadata:
                      sources.append(str(doc.metadata["page"] + 1))


            try:
                for chunk in chain.stream({
                    "question": question,
                    "chat_history": chat_history,
                    "context": context
                }):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")