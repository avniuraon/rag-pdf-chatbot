# RAG PDF Chatbot 🤖

Chat with your PDFs using AI (LangChain + Groq + FAISS)

## 🔥 Demo
<img width="1919" height="911" alt="image" src="https://github.com/user-attachments/assets/e433b225-e2ca-461c-80e1-602c0870d4d3" />


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
