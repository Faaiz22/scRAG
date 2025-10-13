# scRAG - Single-cell RNA-seq Retrieval-Augmented Generation

A Streamlit-based RAG (Retrieval-Augmented Generation) assistant for querying single-cell RNA-seq literature and methods.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd scRAG

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

**Option A: Using the startup script**
```bash
python run_app.py
```

**Option B: Direct Streamlit**
```bash
streamlit run app.py
```

### 3. Using the App

1. **Upload Papers**: Use the sidebar to upload PDF, DOCX, TXT, or MD files
2. **Build Index**: Click "Build Index" to process and embed your documents
3. **Load Index**: Click "Load Index" to load the processed index into memory
4. **Ask Questions**: Enter your questions about single-cell RNA-seq analysis

## 📁 Project Structure

```
scRAG/
├── app.py                          # Main Streamlit application
├── run_app.py                      # Startup script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── papers/                     # Place your PDF/DOCX papers here
│   ├── faiss_index.index          # Generated FAISS index (after building)
│   └── faiss_meta.pkl             # Index metadata (after building)
└── src/
    ├── embeddings.py              # Sentence transformer wrapper
    ├── vectorstore_faiss.py       # FAISS vector store
    ├── ingestion.py               # Document loading (PDF, DOCX, TXT)
    ├── chunking.py                # Text chunking with overlap
    ├── rag_chain.py               # RAG orchestration & LLM wrapper
    ├── utils.py                   # Helper utilities
    └── models/
        ├── supervised.py          # Example: Simple classifier
        └── gan_latent.py          # Example: GAN for latent vectors
```

## 📚 Adding Papers

### Method 1: Through the Streamlit UI
1. Run the app
2. Use the file uploader in the sidebar
3. Click "Save Uploaded Files"

### Method 2: Direct File Copy
```bash
# Copy your papers to the papers directory
cp your_paper.pdf scRAG/data/papers/
```

Supported formats:
- PDF (`.pdf`)
- Word Documents (`.docx`)
- Plain Text (`.txt`)
- Markdown (`.md`)

## 🔧 Configuration

### Using a Local LLM (Optional)

By default, the app uses a mock LLM for generating answers. To use a real HuggingFace model:

```bash
# Set environment variable before running
export HF_MODEL="google/flan-t5-small"  # or any HF model
export HF_MAX_TOKENS=512                # optional
export HF_TEMPERATURE=0.7               # optional
export HF_DEVICE="cuda"                 # optional (cuda/cpu)

# Then run the app
streamlit run app.py
```

### Chunking Parameters

Edit these in `rag_chain.py` if needed:
- `chunk_size`: Default 1200 characters
- `chunk_overlap`: Default 200 characters

### Retrieval Settings

Adjust in the Streamlit sidebar:
- **Top-k retrieval**: Number of relevant chunks to retrieve (1-20)

## 💡 Example Questions

- "What are the best practices for scRNA-seq normalization?"
- "How do I perform clustering on single-cell data?"
- "What models are recommended for cell type annotation?"
- "Explain the steps for differential expression analysis in scRNA-seq"
- "What are highly variable genes and how do I select them?"
- "Compare PCA and t-SNE for dimensionality reduction"

## 🔬 Technical Details

### Embedding Model
- Default: `sentence-transformers/all-mpnet-base-v2`
- Dimension: 768
- Can be changed in `src/embeddings.py`

### Vector Store
- FAISS with L2 distance (IndexFlatL2)
- Stores document metadata alongside vectors
- Persistent storage in `scRAG/data/`

### Document Processing Pipeline
1. **Ingestion**: Load documents from `scRAG/data/papers/`
2. **Chunking**: Split into overlapping chunks
3. **Embedding**: Generate embeddings using sentence-transformers
4. **Indexing**: Store in FAISS with metadata
5. **Retrieval**: Query-time semantic search
6. **Generation**: LLM generates answer from retrieved context

## 🐍 Programmatic Usage

```python
from src.rag_chain import RAGChain
from src.embeddings import Embedder
from src.vectorstore_faiss import FaissStore

# Initialize components
embedder = Embedder()
faiss_store = FaissStore(dim=embedder.embedding_dim())
faiss_store.load()

# Create RAG chain
chain = RAGChain(embedder=embedder, faiss_store=faiss_store)

# Ask a question
answer, hits, distances = chain.answer(
    "How do I normalize scRNA-seq data?",
    k=5
)

print(answer)
```

## 🔨 Rebuilding the Index

If you add new papers or want to change chunking parameters:

1. Add papers to `scRAG/data/papers/`
2. In the Streamlit app, click "Build Index"
3. Wait for processing (may take a few minutes)
4. Click "Load Index" to use the new index

Or programmatically:

```python
from src.rag_chain import RAGChain

chain = RAGChain()
num_chunks = chain.rebuild_index_from_papers()
print(f"Indexed {num_chunks} chunks")
```

## 📊 Model Examples

The `src/models/` directory contains example PyTorch models:

- **supervised.py**: Simple MLP classifier for cell type classification
- **gan_latent.py**: GAN for generating latent representations

These are reference implementations for the types of models discussed in papers.

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set `app.py` as the main file
5. Deploy!

**Note**: Upload papers through the app interface after deployment, or include them in your repo.

### Deploy Locally with Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t scrag .
docker run -p 8501:8501 scrag
```

## 🐛 Troubleshooting

### "No index found"
- Click "Build Index" in the sidebar first
- Make sure you have papers in `scRAG/data/papers/`

### "No documents found"
- Upload papers using the sidebar uploader
- Or copy papers directly to `scRAG/data/papers/`
- Check that files are PDF, DOCX, TXT, or MD format

### Empty or poor quality answers
- The mock LLM provides template responses
- Set `HF_MODEL` environment variable to use a real model
- Try increasing the `k` value for more context
- Ensure your papers contain relevant information

### Memory issues
- Reduce `chunk_size` in the rebuild function
- Use a smaller embedding model
- Process fewer documents at once

## 📝 License

MIT License - feel free to use and modify

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Support

For issues or questions, please open a GitHub issue.

---

**Built with:** Streamlit • FAISS • Sentence-Transformers • PyTorch • HuggingFace
