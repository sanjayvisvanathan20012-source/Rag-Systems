import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import faiss
import numpy as np
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import database

INDEX_FILE = "faiss_index.bin"
EMBED_DIM = 384

class VectorStore:
    def __init__(self, index_file: str = INDEX_FILE, db_path: str = database.DB_PATH):
        self.index_file = index_file
        self.db_path = db_path
        
        # Explicit CPU to prevent Metal / MTLCommandBuffer crashes on macOS
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        database.init_db(self.db_path)
        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> faiss.IndexIDMap2:
        if os.path.exists(self.index_file):
            return faiss.read_index(self.index_file)
        base_index = faiss.IndexFlatIP(EMBED_DIM)
        return faiss.IndexIDMap2(base_index)

    def _save_index(self) -> None:
        faiss.write_index(self.index, self.index_file)

    def get_next_id(self) -> int:
        return self.index.ntotal

    def add_documents(self, chunks: List[Dict]) -> int:
        if not chunks:
            return 0

        texts = [c["text_content"] for c in chunks]
        embeddings = self.embedder.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        start_id = self.get_next_id()
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)

        self.index.add_with_ids(embeddings, ids)
        self._save_index()

        db_records = [
            (int(ids[i]), chunks[i]["filename"], chunks[i]["page_number"], chunks[i]["chunk_index"], chunks[i]["text_content"])
            for i in range(len(chunks))
        ]
        database.insert_chunks(db_records, self.db_path)
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        q_embed = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_embed)

        scores, indices = self.index.search(q_embed, min(top_k, self.index.ntotal))
        matched_ids = [int(idx) for idx in indices[0] if idx != -1]
        
        chunks = database.get_chunks_by_ids(matched_ids, self.db_path)
        for i, c in enumerate(chunks):
            c["score"] = float(scores[0][i])
        return chunks