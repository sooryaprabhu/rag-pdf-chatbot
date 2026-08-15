import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain.schema import Document
from rag import split_documents


def test_split_documents_returns_chunks():
    docs = [Document(
        page_content="This is a test document. " * 100,
        metadata={"source": "test.pdf", "page": 1}
    )]
    chunks = split_documents(docs)
    assert len(chunks) > 0


def test_chunks_not_empty():
    docs = [Document(
        page_content="This is a test document. " * 100,
        metadata={"source": "test.pdf", "page": 1}
    )]
    chunks = split_documents(docs)
    for chunk in chunks:
        assert len(chunk.page_content) > 0


def test_chunk_has_metadata():
    docs = [Document(
        page_content="This is a test document. " * 100,
        metadata={"source": "test.pdf", "page": 1}
    )]
    chunks = split_documents(docs)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test.pdf"


def test_split_returns_list():
    docs = [Document(
        page_content="Short text.",
        metadata={"source": "test.pdf", "page": 1}
    )]
    chunks = split_documents(docs)
    assert isinstance(chunks, list)


def test_multiple_docs():
    docs = [
        Document(
            page_content="Tesla annual report. " * 50,
            metadata={"source": "tesla.pdf", "page": 1}
        ),
        Document(
            page_content="Nvidia annual report. " * 50,
            metadata={"source": "nvidia.pdf", "page": 1}
        )
    ]
    chunks = split_documents(docs)
    assert len(chunks) > 0
