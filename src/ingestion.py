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
            if not os.path.isfile(src):
                continue
            dst = os.path.join(PAPERS_DIR, fname)
            if os.path.abspath(src) != os.path.abspath(dst) and not os.path.exists(dst):
                try:
                    shutil.copy(src, dst)
                    copied.append(fname)
                    print(f"Copied {fname} to {PAPERS_DIR}")
                except Exception as e:
                    print(f"Failed to copy {fname}: {e}")
    return copied

def read_pdf_text(path: str) -> str:
    """Extract text from PDF using pypdf (best-effort)."""
    try:
        reader = PdfReader(path)
        pages = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    pages.append(text)
                else:
                    pages.append("")
            except Exception as e:
                print(f"Error extracting page {page_num} from {path}: {e}")
                pages.append("")
        full_text = "\n".join(pages).strip()
        return full_text
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return ""

def read_docx_text(path: str) -> str:
    """Extract text from DOCX file."""
    try:
        text = docx2txt.process(path)
        return text if text else ""
    except Exception as e:
        print(f"Error reading DOCX {path}: {e}")
        return ""

def read_txt(path: str) -> str:
    """Read plain text file."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {path}: {e}")
        return ""

def load_documents_from_papers_folder(folder: str = None) -> List[Dict]:
    """
    Load supported documents from scRAG/data/papers/.
    Returns list of dicts: { 'source', 'source_path', 'text' }.
    """
    if folder is None:
        folder = PAPERS_DIR
    
    # Ensure folder exists
    ensure_papers_folder()
    
    docs = []
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return docs
    
    print(f"Loading documents from {folder}...")
    
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        
        # Skip directories
        if os.path.isdir(path):
            continue
        
        # Skip hidden files
        if fname.startswith('.'):
            continue
        
        text = ""
        file_type = ""
        
        # Process based on file extension
        if fname.lower().endswith(".pdf"):
            print(f"Processing PDF: {fname}")
            text = read_pdf_text(path)
            file_type = "PDF"
        elif fname.lower().endswith(".docx"):
            print(f"Processing DOCX: {fname}")
            text = read_docx_text(path)
            file_type = "DOCX"
        elif fname.lower().endswith((".txt", ".md")):
            print(f"Processing text file: {fname}")
            text = read_txt(path)
            file_type = "TXT/MD"
        else:
            print(f"Skipping unsupported file: {fname}")
            continue
        
        # Add document if text was extracted
        if text and text.strip():
            doc = {
                "source": fname,
                "source_path": os.path.abspath(path),
                "text": text,
                "file_type": file_type,
                "char_count": len(text)
            }
            docs.append(doc)
            print(f"✓ Loaded {fname} ({file_type}): {len(text)} characters")
        else:
            print(f"✗ No text extracted from {fname}")
    
    print(f"\nTotal documents loaded: {len(docs)}")
    return docs

if __name__ == "__main__":
    copied = ensure_papers_folder()
    print("Copied into scRAG/data/papers:", copied)
    docs = load_documents_from_papers_folder()
    print(f"\nFound {len(docs)} documents in scRAG/data/papers/")
    for doc in docs:
        print(f"  - {doc['source']}: {doc['char_count']} chars")
