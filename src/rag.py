import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


def load_pdf(filepath):
    # Read the PDF file and extract all text
    # pypdf opens the PDF page by page
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(f"PDF loaded: {len(text)} characters extracted")
    return text


def split_text(text):
    # Split text into small chunks
    # We can't send the whole document to OpenAI at once
    # chunk_size = max characters per chunk
    # chunk_overlap = how many characters overlap between chunks
    # overlap helps preserve context at chunk boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    # Convert chunks to embeddings and store in FAISS
    # OpenAIEmbeddings converts text to vectors
    # FAISS stores and indexes those vectors
    embeddings = OpenAIEmbeddings(
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    print("Vector store created successfully")
    return vector_store


def create_qa_chain(vector_store):
    # Create the question answering chain
    # This connects:
    # retriever (finds relevant chunks)
    # + LLM (answers based on those chunks)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    # retriever searches the vector store
    # k=3 means find top 3 most relevant chunks
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    # RetrievalQA combines retriever + LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )
    print("QA chain ready!")
    return qa_chain


def ask_question(qa_chain, question):
    # Ask a question and get an answer
    print(f"\nQuestion: {question}")
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    print(f"Answer: {answer}")
    return answer


if __name__ == "__main__":
    # Test the full RAG pipeline
    pdf_path = "data/📍Soorya_Prabhu_CV.pdf"

    # Step 1 — Load PDF
    text = load_pdf(pdf_path)

    # Step 2 — Split into chunks
    chunks = split_text(text)

    # Step 3 — Create vector store
    vector_store = create_vector_store(chunks)

    # Step 4 — Create QA chain
    qa_chain = create_qa_chain(vector_store)

    # Step 5 — Ask questions about YOUR CV
    ask_question(qa_chain, "What is Soorya's educational background?")
    ask_question(qa_chain, "What programming languages does Soorya know?")
    ask_question(qa_chain, "What projects has Soorya worked on?")
