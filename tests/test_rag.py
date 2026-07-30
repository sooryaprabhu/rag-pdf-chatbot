import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag import load_pdf, split_text


# ── TEST 1 ────────────────────────────────────────────────────────────
# Test that split_text returns chunks
def test_split_text_returns_chunks():
    sample_text = "This is a test document. " * 100
    chunks = split_text(sample_text)
    assert len(chunks) > 0, "Should return at least one chunk"


# ── TEST 2 ────────────────────────────────────────────────────────────
# Test that chunks are not empty
def test_chunks_not_empty():
    sample_text = "This is a test document. " * 100
    chunks = split_text(sample_text)
    for chunk in chunks:
        assert len(chunk) > 0, "No chunk should be empty"


# ── TEST 3 ────────────────────────────────────────────────────────────
# Test that chunk size is within limits
def test_chunk_size_limit():
    sample_text = "This is a test document. " * 100
    chunks = split_text(sample_text)
    for chunk in chunks:
        assert len(chunk) <= 600, "Chunks should not exceed size limit"


# ── TEST 4 ────────────────────────────────────────────────────────────
# Test that text splitting handles short text
def test_split_short_text():
    sample_text = "Short text."
    chunks = split_text(sample_text)
    assert len(chunks) >= 1, "Should handle short text"


# ── TEST 5 ────────────────────────────────────────────────────────────
# Test that split_text returns a list
def test_split_returns_list():
    sample_text = "This is a test document. " * 50
    chunks = split_text(sample_text)
    assert isinstance(chunks, list), "Should return a list"
