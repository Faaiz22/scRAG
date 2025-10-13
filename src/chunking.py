"""
scRAG/src/chunking.py

Character-based chunking for long documents. Keeps metadata linking to sources.
"""
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= n:
            break
        start = max(0, end - chunk_overlap)
    return chunks

def docs_to_chunks(docs: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    out = []
    cid = 0
    for d in docs:
        txt = d.get("text", "") or ""
        chs = chunk_text(txt, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c in chs:
            out.append({
                "id": f"chunk-{cid}",
                "source": d.get("source"),
                "source_path": d.get("source_path"),
                "text": c,
                "excerpt": c[:300].replace("\n", " ")
            })
            cid += 1
    return out
