import sqlite3
from typing import List, Dict, Optional, Tuple

DB_PATH = "rag_metadata.db"

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                vector_id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text_content TEXT NOT NULL
            );
        """)
        conn.commit()

def insert_chunks(chunks_metadata: List[Tuple[int, str, int, int, str]], db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO chunks (vector_id, filename, page_number, chunk_index, text_content)
            VALUES (?, ?, ?, ?, ?);
        """, chunks_metadata)
        conn.commit()

def get_chunk_by_id(vector_id: int, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE vector_id = ?;", (vector_id,))
        row = cur.fetchone()
        return dict(row) if row else None

def get_chunks_by_ids(vector_ids: List[int], db_path: str = DB_PATH) -> List[Dict]:
    if not vector_ids:
        return []
    placeholders = ",".join("?" for _ in vector_ids)
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM chunks WHERE vector_id IN ({placeholders});", vector_ids)
        rows = cur.fetchall()
        id_map = {row["vector_id"]: dict(row) for row in rows}
        return [id_map[vid] for vid in vector_ids if vid in id_map]

def get_document_stats(db_path: str = DB_PATH) -> List[Dict]:
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT filename, COUNT(*) as chunk_count, MAX(page_number) as max_pages 
            FROM chunks 
            GROUP BY filename;
        """)
        return [dict(row) for row in cur.fetchall()]
