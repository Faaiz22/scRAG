# scRAG - Single-cell Retrieval-Augmented Generation (demo)

Small reproducible project demonstrating a RAG assistant for single-cell RNA-seq literature and modeling.

## Quick steps (Colab)
1. Open a fresh Google Colab.
2. Run the provided Colab cells in order:
   - Install dependencies and write repo files
   - Upload PDFs into `scRAG/data/papers/` (Colab sidebar or copy from /mnt/data)
   - Build FAISS index (ingest -> chunk -> embed -> save)
   - (Optional) Set `HF_MODEL` env var to use a local HF model for generation
   - Run sample queries and push repo to GitHub

See `notebooks/colab_demo_cells.txt` for explicit cells.

## Notes
- The demo uses a mock LLM by default. Set `HF_MODEL` to a HF model name (e.g., `google/flan-t5-small`) to use a local model.
- Streamlit app at `scRAG/app.py`. Deploy to Streamlit Cloud after pushing repo.

