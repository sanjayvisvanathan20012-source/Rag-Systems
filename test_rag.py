import os
import pytest
from chunking import sliding_window_chunk, process_document_pages
from ingestion import parse_txt
from vector_store import VectorStore
import database

TEST_DB = "test_metadata.db"
TEST_INDEX = "test_faiss.bin"

@pytest.fixture(autouse=True)
def clean_test_env():
    for f in [TEST_DB, TEST_INDEX]:
        if os.path.exists(f):
            os.remove(f)
    yield
    for f in [TEST_DB, TEST_INDEX]:
        if os.path.exists(f):
            os.remove(f)

def test_sliding_window_chunk():
    text = "A" * 3000
    chunks = sliding_window_chunk(text, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 4
    assert len(chunks[0]) == 1000

def test_txt_parsing_and_chunking(tmp_path):
    sample_file = tmp_path / "handbook.txt"
    sample_file.write_text("University Policy: Students must maintain a 3.0 GPA for honors eligibility.")
    
    pages = parse_txt(str(sample_file))
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1

    chunks = process_document_pages(pages, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert "honors eligibility" in chunks[0]["text_content"]

def test_vector_store_add_and_retrieve():
    store = VectorStore(index_file=TEST_INDEX, db_path=TEST_DB)
    chunks = [
        {"filename": "cs_syllabus.pdf", "page_number": 3, "chunk_index": 0, "text_content": "CS101 covers Turing machines and computational complexity."},
        {"filename": "admin_guide.pdf", "page_number": 12, "chunk_index": 1, "text_content": "Tuition payments are due by the 15th of September annually."}
    ]
    added = store.add_documents(chunks)
    assert added == 2

    results = store.search("computational complexity", top_k=1)
    assert len(results) == 1
    assert results[0]["filename"] == "cs_syllabus.pdf"
    assert results[0]["page_number"] == 3
