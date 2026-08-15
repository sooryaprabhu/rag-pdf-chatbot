import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
import time


def get_pinecone_index(index_name="rag-pdf-chatbot"):
    api_key = os.environ.get('PINECONE_API_KEY')
    pc = Pinecone(api_key=api_key)
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
        print(f"Created index: {index_name}")
    return pc.Index(index_name)


def load_pdf(filepath, filename="document"):
    reader = PdfReader(filepath)
    documents = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            documents.append(Document(
                page_content=page_text,
                metadata={"source": filename, "page": i + 1}
            ))
    print(f"Loaded: {filename} — {len(documents)} pages")
    return "", documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, index_name="rag-pdf-chatbot"):
    api_key = os.environ.get('PINECONE_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')

    embeddings = OpenAIEmbeddings(api_key=openai_key)

    # Get or create index
    get_pinecone_index(index_name)

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=api_key
    )

    # Add documents
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vector_store.add_texts(texts, metadatas=metadatas)

    print(f"Stored {len(chunks)} chunks!")
    return vector_store


def add_pdf_to_store(vector_store, chunks):
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vector_store.add_texts(texts, metadatas=metadatas)
    print(f"Added {len(chunks)} more chunks!")
    return vector_store


def create_qa_chain(vector_store, uploaded_files=None):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.environ.get('OPENAI_API_KEY')
    )

    files_context = ""
    if uploaded_files:
        files_context = f"The uploaded documents are: {', '.join(uploaded_files)}. "

    prompt_template = f"""You are a helpful financial analyst assistant.
{files_context}
ALWAYS mention the document name and company when answering.
Never say "the company" — always use the actual company name.

Context: {{context}}
Question: {{question}}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def ask_with_sources(qa_chain, question):
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])
    return answer, sources
