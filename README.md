# RAG PDF Chatbot

Upload any PDF and ask questions about it using RAG and OpenAI.

## Tech Stack
- LangChain
- FAISS vector database
- OpenAI GPT-4o-mini
- FastAPI
- Docker

## How it works
1. Upload a PDF via /upload-pdf
2. PDF is split into chunks
3. Chunks converted to embeddings
4. Stored in FAISS vector database
5. Ask questions via /ask
6. RAG retrieves relevant chunks
7. GPT-4o-mini answers using those chunks

## Run locally
pip install -r requirements.txt
uvicorn src.api:app --reload

## API Endpoints
- GET  /              health check
- POST /upload-pdf    upload a PDF
- POST /ask           ask a question

## Author
Soorya Prabhu - MSc Artificial Intelligence, Brunel University London
