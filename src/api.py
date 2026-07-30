from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rag import load_pdf, split_text, create_vector_store, create_qa_chain

app = FastAPI(title="RAG PDF Chatbot API")

# Global variable to store the QA chain
# Once PDF is uploaded we keep it in memory
qa_chain = None


class Question(BaseModel):
    question: str


@app.get("/")
def health_check():
    return {"status": "RAG PDF Chatbot is running!"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain

    # Save uploaded PDF to a temporary file
    # tempfile creates a temporary location on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Run the full RAG pipeline on the uploaded PDF
    text = load_pdf(tmp_path)
    chunks = split_text(text)
    vector_store = create_vector_store(chunks)
    qa_chain = create_qa_chain(vector_store)

    # Clean up temporary file
    os.unlink(tmp_path)

    return {
        "message": "PDF uploaded and processed successfully",
        "chunks_created": len(chunks),
        "characters_extracted": len(text)
    }


@app.post("/ask")
def ask_question(body: Question):
    global qa_chain

    # Check if PDF has been uploaded first
    if qa_chain is None:
        return {
            "error": "No PDF uploaded yet. Please upload a PDF first."
        }

    # Get answer from RAG pipeline
    result = qa_chain.invoke({"query": body.question})
    answer = result["result"]

    return {
        "question": body.question,
        "answer": answer
    }
