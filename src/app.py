import streamlit as st
import tempfile
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from rag import load_pdf, split_documents, create_vector_store, add_pdf_to_store, create_qa_chain, ask_with_sources

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG PDF Chatbot")
st.markdown("Upload multiple PDFs and ask questions across all of them")
st.markdown("Powered by **LangChain** + **Pinecone** + **OpenAI**")
st.divider()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

uploaded_files = st.file_uploader(
    "Upload your PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDFs — annual reports, research papers, contracts"
)

if uploaded_files:
    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state.uploaded_pdfs
    ]

    if new_files:
        for uploaded_file in new_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                text, documents = load_pdf(
                    tmp_path,
                    filename=uploaded_file.name
                )
                chunks = split_documents(documents)

                if st.session_state.vector_store is None:
                    st.session_state.vector_store = create_vector_store(chunks)
                else:
                    add_pdf_to_store(st.session_state.vector_store, chunks)

                st.session_state.uploaded_pdfs.append(uploaded_file.name)

                # Pass ALL uploaded file names to the chain
                st.session_state.qa_chain = create_qa_chain(
                    st.session_state.vector_store,
                    uploaded_files=st.session_state.uploaded_pdfs
                )
                os.unlink(tmp_path)

            st.success(f"✅ {uploaded_file.name} processed!")

if st.session_state.uploaded_pdfs:
    st.info(f"📚 Loaded: {', '.join(st.session_state.uploaded_pdfs)}")

if st.button("🗑️ Clear all PDFs and start over"):
    st.session_state.qa_chain = None
    st.session_state.messages = []
    st.session_state.uploaded_pdfs = []
    st.session_state.vector_store = None
    st.rerun()

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Source excerpts"):
                for i, src in enumerate(msg["sources"][:3]):
                    source_name = src.metadata.get("source", "Unknown")
                    page_num = src.metadata.get("page", "?")
                    st.markdown(f"**📄 {source_name} — Page {page_num}:**")
                    st.markdown(f"> {src.page_content[:300]}...")
                    st.divider()

if st.session_state.qa_chain is not None:
    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Searching across all PDFs..."):
                answer, sources = ask_with_sources(
                    st.session_state.qa_chain,
                    question
                )
                st.write(answer)

                if sources:
                    with st.expander("📚 Source excerpts"):
                        for i, src in enumerate(sources[:3]):
                            source_name = src.metadata.get("source", "Unknown")
                            page_num = src.metadata.get("page", "?")
                            st.markdown(
                                f"**📄 {source_name} — Page {page_num}:**"
                            )
                            st.markdown(f"> {src.page_content[:300]}...")
                            st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
else:
    st.info("👆 Please upload at least one PDF to start!")

st.divider()
st.markdown(
    "Built by **Soorya Prabhu** | "
    "MSc AI, Brunel University London | "
    "LangChain + Pinecone + OpenAI | "
    "[GitHub](https://github.com/sooryaprabhu/rag-pdf-chatbot)"
)
