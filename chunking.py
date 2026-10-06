from typing import List, Dict

def sliding_window_chunk(
    text: str, 
    chunk_size: int = 1500, 
    chunk_overlap: int = 200
) -> List[str]:
    text = text.strip()
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start += step

    return chunks

def process_document_pages(
    pages_data: List[Dict], 
    chunk_size: int = 1500, 
    chunk_overlap: int = 200
) -> List[Dict]:
    processed_chunks = []
    chunk_index = 0
    for page in pages_data:
        raw_chunks = sliding_window_chunk(
            page["text"], 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        for chunk in raw_chunks:
            processed_chunks.append({
                "filename": page["filename"],
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text_content": chunk
            })
            chunk_index += 1
    return processed_chunks
