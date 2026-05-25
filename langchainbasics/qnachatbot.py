import streamlit as st
import os
import uuid

# Core LangChain & Partner Imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Document Parsing & Vector Architecture
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The rest of your NexusRAG code remains exactly the same...
# -------- UI CONFIG & STYLING --------
st.set_page_config(
    page_title="PDF CHATBOT", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom injection for a clean, professional dark developer aesthetic
st.markdown("""
    <style>
        .main { background-color: #0F111A; }
        div[data-testid="stSidebarUserContent"] { background-color: #161925; }
        .stChatMessage { border-radius: 8px; margin-bottom: 12px; }
        .stChatMessage[data-testid="stChatMessageUser"] { background-color: #1E2235; border: 1px solid #303651; }
        .stChatMessage[data-testid="stChatMessageAssistant"] { background-color: #161925; border: 1px solid #252A3F; }
        .metric-card {
            background-color: #161925;
            border: 1px solid #252A3F;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value { font-size: 24px; font-weight: bold; color: #4F46E5; }
        .metric-label { font-size: 12px; color: #8A92B2; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

st.title("PDF CHATBOT")
st.caption("Multi-Document Agentic Engine powered by Groq Llama-3.1")

# -------- SIDEBAR CONTROLS --------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/64/artificial-intelligence.png", width=50)
    st.header("System Controls")
    
    api_key = st.text_input("GROQ API Key", type="password", help="Enter a valid Groq API key. You can obtain one from your Groq account dashboard.")
    
    model_name = st.selectbox(
        "Inference Engine",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "gemma2-9b-it"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Upload PDF here")
    # Multi-file upload enabled
    uploaded_files = st.file_uploader(
        "Upload Source Documents (PDF)", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload one or multiple documents to build your context index."
    )
    
    st.markdown("---")
    st.subheader(" Tuning Parameters")
    chunk_size = st.slider("Chunk Size Tokens", 200, 2000, 800, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 150, step=50)
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.2, step=0.1)

    if st.button(" Reset Global Cache", use_container_width=True):
        st.session_state.messages = []
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]
        st.rerun()

# -------- SESSION STATE MEMORY --------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------- ARCHITECTURAL PIPELINES --------

@st.cache_resource(show_spinner=False)
def ingest_multiple_pdfs(files, _chunk_size, _chunk_overlap):
    """Processes multiple uploaded PDFs, builds uniform metadata, and compiles a single vector index."""
    all_chunks = []
    processed_manifest = []
    
    # Create temporary directory if needed
    os.makedirs("temp_storage", exist_ok=True)
    
    for uploaded_file in files:
        file_id = str(uuid.uuid4())[:8]
        temp_path = os.path.join("temp_storage", f"{file_id}_{uploaded_file.name}")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
            
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        # Inject structural metadata into individual files before chunking
        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        
        processed_manifest.append({"name": uploaded_file.name, "pages": len(docs), "chunks": len(chunks)})
        
        # Clean up disk space immediately after load
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    
    return vectorstore, processed_manifest

# Compile and store database reference dynamically inside runtime session memory
retriever = None
manifest = []
if uploaded_files:
    with st.spinner("🔄 Indexing Knowledge Base..."):
        vectorstore, manifest = ingest_multiple_pdfs(uploaded_files, chunk_size, chunk_overlap)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -------- MULTI-TURN QUERY TRANSFORMATION ENGINE --------
def transform_query(raw_query, memory_history, llm):
    """
    Executes a Query Transformation Step. Evaluates conversation logs to translate
    ambiguous references into clean, absolute, database-optimized search strings.
    """
    if not memory_history:
        return raw_query
        
    transformation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an advanced query transformation agent. "
                   "Analyze the chat history and the latest question to output an optimized, "
                   "stand-alone keyword search query for a document index database. "
                   "Do NOT add preamble, commentary, or pleasantries. Output ONLY the raw optimized string."),
        ("placeholder", "{chat_history}"),
        ("user", "Generate optimized search query for: {query}")
    ])
    
    transformation_chain = transformation_prompt | llm | StrOutputParser()
    try:
        return transformation_chain.invoke({"query": raw_query, "chat_history": memory_history}).strip()
    except Exception:
        return raw_query

# -------- LLM CORES --------
def initialize_llm(api_key, model, temp):
    return ChatGroq(groq_api_key=api_key, model_name=model, temperature=temp, streaming=True)

def compile_generation_chain(api_key, model, temp):
    if not api_key:
        return None
    llm = initialize_llm(api_key, model, temp)
    
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an elite enterprise technical consultant. "
                   "Answer questions accurately using ONLY the structured context provided below. "
                   "Cite the source file names and pages naturally when explaining your findings.\n\n"
                   "Context Database:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("user", "{question}")
    ])
    return system_prompt | llm | StrOutputParser()

# Main orchestration variable binding
engine_llm = initialize_llm(api_key, model_name, temperature) if api_key else None
generation_chain = compile_generation_chain(api_key, model_name, temperature)

# -------- LIVE ARCHITECTURE LAYOUT --------
if not api_key:
    st.info("🔑 Please initialize system by inputting your GROQ API Key in the sidebar dashboard panel.")
else:
    # Knowledge Space Visual Dashboards (Upgraded UI Metrics)
    if manifest:
        cols = st.columns(len(manifest) if len(manifest) <= 4 else 4)
        for idx, item in enumerate(manifest[:4]):
            with cols[idx]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{item['chunks']}</div>
                        <div class="metric-label">📄 {item['name'][:18]}... ({item['pages']} pgs)</div>
                    </div>
                """, unsafe_allow_html=True)
        st.write("") # Spacer

    # Message Renderer Loop
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                st.caption(f"🛡️ **Verified Grounded Context:** {msg['sources']}")

    # Real-Time Execution Framework
    if question := st.chat_input("Query aggregated file knowledge space..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Recompile clean session dialogue snapshots
        chat_history = [
            ("user" if m["role"] == "user" else "assistant", m["content"])
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            response_box = st.empty()
            full_response = ""
            formatted_source_footer = ""

            context_stream = ""
            if retriever:
                # 1. Execute Query Transformation Pipeline Step
                transformed_search = transform_query(question, chat_history, engine_llm)
                
                # Modern Diagnostics Window
                with st.expander("🛠️ Advanced Execution Diagnostics", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**User Intent Signature:**")
                        st.caption(f"`{question}`")
                    with col2:
                        st.markdown("**Transformed Database Query:**")
                        st.code(transformed_search, language="text")

                # 2. Extract relative chunks
                retrieved_chunks = retriever.invoke(transformed_search)
                
                # Compile standard document layout mappings
                context_blocks = []
                unique_source_references = set()
                
                for doc in retrieved_chunks:
                    fname = doc.metadata.get("source_file", "Unknown Doc")
                    pnum = doc.metadata.get("page", 0) + 1
                    context_blocks.append(f"--- [File: {fname} | Page: {pnum}] ---\n{doc.page_content}")
                    unique_source_references.add(f"{fname} (Pg. {pnum})")
                    
                context_stream = "\n\n".join(context_blocks)
                if unique_source_references:
                    formatted_source_footer = " | ".join(sorted(list(unique_source_references)))

            # 3. Stream Inference generation pass-through
            try:
                for chunk in generation_chain.stream({
                    "question": question,
                    "chat_history": chat_history,
                    "context": context_stream
                }):
                    full_response += chunk
                    response_box.markdown(full_response + "▌")

                response_box.markdown(full_response)
                
                if formatted_source_footer:
                    st.caption(f"🛡️ **Verified Grounded Context:** {formatted_source_footer}")

                # Save entire frame state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": formatted_source_footer
                })

            except Exception as e:
                st.error(f"Inference Loop Failure: {str(e)}")