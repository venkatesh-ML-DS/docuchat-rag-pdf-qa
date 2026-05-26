
# DocuChat — RAG-based PDF Q&A System

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions. The system retrieves relevant content from the document and generates accurate answers using Google Gemini AI.

## Features
- Upload any PDF document
- Ask natural language questions about the document
- Semantic search using FAISS vector store
- AI-powered answers using Google Gemini 2.0 Flash

## Tech Stack
- **Backend:** Python, Flask
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Vector Store:** FAISS
- **LLM:** Google Gemini 2.0 Flash
- **PDF Processing:** PyPDF

## Installation

```bash
git clone https://github.com/venkatesh-ML-DS/docuchat-rag-pdf-qa.git
cd docuchat-rag-pdf-qa
pip install -r requirements.txt
```

## Setup
Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

## Run Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## Future Improvements
- Multi-PDF support
- Chat history
- Better UI/UX
- Cloud deployment

## Author
K M Venkatesh  
GitHub: https://github.com/venkatesh-ML-DS
