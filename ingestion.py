import os
from typing import List, Dict
import fitz
import docx

def parse_pdf(file_path: str) -> List[Dict]:
    filename = os.path.basename(file_path)
    pages_data = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        if text:
            pages_data.append({
                "filename": filename,
                "page_number": page_num + 1,
                "text": text
            })
    doc.close()
    return pages_data

def parse_docx(file_path: str) -> List[Dict]:
    filename = os.path.basename(file_path)
    doc = docx.Document(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [{
        "filename": filename,
        "page_number": 1,
        "text": full_text
    }] if full_text else []

def parse_txt(file_path: str) -> List[Dict]:
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return [{
        "filename": filename,
        "page_number": 1,
        "text": text
    }] if text else []

def parse_file(file_path: str) -> List[Dict]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
