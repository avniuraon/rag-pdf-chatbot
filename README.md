# RAG PDF Chatbot 🤖

Chat with your PDFs using AI (LangChain + Groq + FAISS)

## 🔥 Demo
(Add screenshot or gif here)

## 🚀 Features
- Upload PDF and ask questions
- Context-aware answers using RAG
- Streaming responses
- Source page extraction

## 🧠 How it works
1. PDF → text chunks
2. Embeddings → FAISS vector DB
3. Query → retrieve relevant chunks
4. LLM generates answer

## 🛠 Tech Stack
- Python
- LangChain
- Streamlit
- FAISS
- Groq API

## ▶️ Run locally
pip install -r requirements.txt  
streamlit run main.py
