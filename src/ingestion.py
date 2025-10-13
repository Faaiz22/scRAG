"""
scRAG/src/ingestion.py

Robust loaders that explicitly read from scRAG/data/papers/,
copy PDFs from /mnt/data or /content into that folder, and extract text.
"""
import os
import shutil
from typing import List, Dict
from pypdf import PdfReader
import docx2txt

PAPERS_DIR = os.path.join("scRAG", "data", "papers")

def ensure_papers_folder():
    """Create papers folder and copy PDFs from common upload locations into it."""
    os.makedirs(PAPERS_DIR, exist_ok=True)
    copied = []
    candidates = ["/mnt/data", "/content", "."]
    for root in candidates:
        if not os.path.exists(root):
            continue
        for fname in os.listdir(root):
            if not fname.lower().endswith((".pdf", ".docx", ".txt", ".md")):
                continue
            src = os.path.join(root, fname)
            dst = os.path.join(PAPERS_DIR, fname)
            if os.path.abspath(src) != os.path.abspath(dst) and not os.path.exists(dst):
                try:
                    shutil.copy(src, dst)
                    copied.append(fname)
                except Exception:
                    pass
    return copied

def read_pdf_text(path: str) -> str:
    """Extract text from PDF using pypdf (best-effort)."""
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    except Exception:
        return ""

def read_docx_text(path: str) -> str:
    try:
        return docx2txt.process(path) or ""
    except Exception:
        return ""

def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def load_documents_from_papers_folder(folder: str = None) -> List[Dict]:
    """
    Load supported documents from scRAG/data/papers/.
    Returns list of dicts: { 'source', 'source_path', 'text' }.
    """
    if folder is None:
        folder = PAPERS_DIR
    ensure_papers_folder()
    docs = []
    if not os.path.exists(folder):
        return docs
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if os.path.isdir(path):
            continue
        text = ""
        if fname.lower().endswith(".pdf"):
            text = read_pdf_text(path)
        elif fname.lower().endswith(".docx"):
            text = read_docx_text(path)
        elif fname.lower().endswith((".txt", ".md")):
            text = read_txt(path)
        else:
            continue
        if text and text.strip():
            docs.append({"source": fname, "source_path": os.path.abspath(path), "text": text})
    return docs

if __name__ == "__main__":
    copied = ensure_papers_folder()
    print("Copied into scRAG/data/papers:", copied)
    docs = load_documents_from_papers_folder()
    print(f"Found {len(docs)} documents in scRAG/data/papers/")
