import os
import re
from typing import List, Tuple
import pdfminer.high_level
import docx2txt

def parse_pdf(path: str) -> str:
    return pdfminer.high_level.extract_text(path)

def parse_docx(path: str) -> str:
    return docx2txt.process(path) or ""

def parse_txt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

EXTENSION_TO_PARSER = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_txt,
}

def sanitize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def chunk_text(text: str, max_tokens: int = 500, min_tokens: int = 300) -> List[str]:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for s in sentences:
        s = s.strip()
        if not s: continue
        candidate = (current_chunk + " " + s).strip() if current_chunk else s
        n_tokens = len(enc.encode(candidate))
        if n_tokens < min_tokens:
            current_chunk = candidate
        elif n_tokens > max_tokens:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = s
        else:
            current_chunk = candidate
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
