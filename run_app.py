"""
scRAG/run_app.py - Startup script for the scRAG Streamlit app
"""
import os
import sys
import subprocess

def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        "scRAG/data",
        "scRAG/data/papers",
        "scRAG/src",
        "scRAG/src/models"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Directory: {d}")

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'streamlit',
        'faiss',
        'sentence_transformers',
        'torch',
        'pypdf',
        'docx2txt'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'faiss':
                import faiss
            elif package == 'sentence_transformers':
                import sentence_transformers
            elif package == 'torch':
                import torch
            elif package == 'pypdf':
                import pypdf
            elif package == 'docx2txt':
                import docx2txt
            elif package == 'streamlit':
                import streamlit
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies installed")
    return True

def main():
    print("=" * 60)
    print("scRAG - Single-cell RNA-seq RAG Assistant")
    print("=" * 60)
    print()
    
    # Check directory structure
    print("Checking directory structure...")
    ensure_directories()
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    print()
    
    # Check for existing index
    index_path = "scRAG/data/faiss_index.index"
    meta_path = "scRAG/data/faiss_meta.pkl"
    
    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("✓ Existing index found")
    else:
        print("ℹ️  No index found. You'll need to build one in the app.")
    print()
    
    # Start Streamlit
    print("=" * 60)
    print("Starting Streamlit app...")
    print("=" * 60)
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\nApp stopped by user.")
    except Exception as e:
        print(f"\n\n❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() papers
    papers_dir = "scRAG/data/papers"
    papers = [f for f in os.listdir(papers_dir) if f.lower().endswith(('.pdf', '.docx', '.txt', '.md'))]
    
    if papers:
        print(f"✓ Found {len(papers)} paper(s) in {papers_dir}")
        for p in papers:
            print(f"  - {p}")
    else:
        print(f"⚠️  No papers found in {papers_dir}")
        print(f"   Upload papers through the Streamlit interface or add them to this folder.")
    print()
    
    # Check for