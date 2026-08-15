import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec


def get_pinecone_client():
    api_key = os.environ.get('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found!")
    return Pinecone(api_key=api_key)


def load_pdf(filepath, filename="document"):
    reader = PdfReader(filepath)
    documents = []
    full_text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text += page_text
            documents.append(Document(
                page_content=page_text,
                metadata={
                    "source": filename,
                    "page": i + 1
                }
            ))
    print(f"PDF loaded: {filename} — {len(documents)} pages")
    return full_text, documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, index_name="rag-pdf-chatbot"):
    pc = get_pinecone_client()

    existing = [i.name for i in pc.list_indexes()]

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created Pinecone index: {index_name}")
    else:
        print(f"Using existing index: {index_name}")

    embeddings = OpenAIEmbeddings(
        api_key=os.environ.get('OPENAI_API_KEY')
    )

    vector_store = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name,
        pinecone_api_key=os.environ.get('PINECONE_API_KEY')
    )
    print(f"Stored {len(chunks)} chunks in Pinecone!")
    return vector_store


def add_pdf_to_store(vector_store, chunks):
    embeddings = OpenAIEmbeddings(
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    vector_store.add_documents(chunks)
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
Each piece of context below comes from a specific document.
ALWAYS mention the document name and company when answering.
If comparing multiple companies clearly label each company.
Never say "the company" — always use the actual company name.

Context: {{context}}
Question: {{question}}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

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
