🧠 NexusRAG | AI Knowledge Hub

An advanced RAG (Retrieval-Augmented Generation) system built with Streamlit, LangChain, ChromaDB, and Groq (Llama-3.3-70B). NexusRAG supports hybrid retrieval using BM25 keyword search and Semantic Vector Search combined via Reciprocal Rank Fusion (RRF).

✨ Features

🔍 Hybrid Search Engine: Combines BM25 lexical search with semantic embeddings for maximum retrieval accuracy.

🔀 Reciprocal Rank Fusion (RRF): Merges sparse and dense search results dynamically.

⚡ Groq Powered Generation: Uses ultra-fast Llama 3.3 70B Versatile LLM.

🌐 Multilingual Support: Arabic and English query understanding with accurate context synthesis.

📄 Multi-Format Upload: Supports indexing multiple PDF and TXT documents.

🗑️ Granular Document Management: Delete specific files from the vector database on the fly without breaking the index.

🎨 Modern Sleek UI: Fully dark-mode styled high-contrast Streamlit interface.

📁 Project Structure

nexus-rag/
├── app.py                   # Streamlit web interface
├── src/
│   ├── config.py            # Configuration and environment variables
│   ├── document_loader.py   # PDF/TXT parsing & text chunking
│   ├── embeddings.py        # Embedding model loader
│   ├── vector_store.py      # ChromaDB vector store & Hybrid RRF search
│   └── rag_engine.py        # Groq LLM integration & prompt template
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore setup
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation


🚀 Quick Start Guide

1. Clone the Repository

git clone https://github.com/your-username/nexus-rag.git
cd nexus-rag


2. Create and Activate Virtual Environment

# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Configure Environment Variables

Create a .env file in the root directory based on .env.example:

GROQ_API_KEY=gsk_your_actual_groq_api_key_here


5. Run the Application

streamlit run app.py


🛠️ Built With

Streamlit

LangChain

Groq AI

ChromaDB

HuggingFace Sentence Transformers

📜 License

This project is licensed under the MIT License.