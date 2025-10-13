"""
scRAG/src/rag_chain.py

RAG orchestration: rebuild index helper, retrieval, and LLM wrapper.
Supports optional local HF model via HF_MODEL env var; otherwise uses mock.
"""
import os
from typing import List, Tuple
from src.embeddings import Embedder
from src.vectorstore_faiss import FaissStore
from src.chunking import docs_to_chunks
from src.ingestion import load_documents_from_papers_folder

# Optional HF support
HF_AVAILABLE = False
try:
    from transformers import pipeline
    import torch
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

class LLMWrapper:
    def __init__(self, hf_env_key: str = "HF_MODEL"):
        self.hf_model_name = os.environ.get(hf_env_key, None)
        self.hf_device = os.environ.get("HF_DEVICE", None)
        self.max_tokens = int(os.environ.get("HF_MAX_TOKENS", 256))
        self.temperature = float(os.environ.get("HF_TEMPERATURE", 0.0))
        self.pipeline = None
        self.is_seq2seq = False
        
        if self.hf_model_name and HF_AVAILABLE:
            print(f"Attempting to load HF model: {self.hf_model_name}")
            try:
                device = 0 if (self.hf_device == "cuda" or (self.hf_device is None and torch.cuda.is_available())) else -1
                try:
                    self.pipeline = pipeline("text2text-generation", model=self.hf_model_name, device=device)
                    self.is_seq2seq = True
                    print(f"✓ Loaded seq2seq model on device {device}")
                except Exception:
                    self.pipeline = pipeline("text-generation", model=self.hf_model_name, device=device)
                    self.is_seq2seq = False
                    print(f"✓ Loaded text-generation model on device {device}")
            except Exception as e:
                print(f"✗ Failed to load HF model {self.hf_model_name}: {e}")
                print("Falling back to mock LLM")
                self.pipeline = None
        else:
            if not self.hf_model_name:
                print("No HF_MODEL environment variable set. Using mock LLM.")
            elif not HF_AVAILABLE:
                print("HuggingFace transformers not available. Using mock LLM.")

    def generate(self, question: str, contexts: List[dict]) -> str:
        """Generate answer based on question and retrieved contexts."""
        # Build context block from retrieved documents
        ctx_lines = []
        for i, c in enumerate(contexts[:8]):
            if not c:
                continue
            title = c.get("source", "unknown")
            excerpt = (c.get("excerpt") or c.get("text", "")[:400]).replace("\n", " ")
            ctx_lines.append(f"[{i+1}] {title} -- {excerpt[:400]}")
        
        context_block = "\n".join(ctx_lines) if ctx_lines else "No relevant context found."
        
        # Build prompt
        prompt = (
            "You are a helpful assistant specializing in single-cell genomics and RNA-seq analysis.\n"
            "Use the retrieved evidence below to answer the user's question accurately and concisely.\n\n"
            "Retrieved evidence:\n"
            f"{context_block}\n\n"
            "Instructions:\n"
            "1. Answer based on the evidence provided\n"
            "2. Be specific and cite sources using [n] notation\n"
            "3. If asked about methods, provide step-by-step instructions\n"
            "4. For model recommendations, include architecture and hyperparameters\n"
            "5. If the evidence doesn't contain the answer, say so honestly\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        # Generate using HF model if available
        if self.pipeline is not None:
            try:
                if self.is_seq2seq:
                    out = self.pipeline(
                        prompt,
                        max_length=self.max_tokens,
                        do_sample=(self.temperature > 0.0),
                        temperature=self.temperature if self.temperature > 0.0 else None
                    )
                    return out[0]["generated_text"].strip()
                else:
                    out = self.pipeline(
                        prompt,
                        max_new_tokens=self.max_tokens,
                        do_sample=(self.temperature > 0.0),
                        temperature=self.temperature if self.temperature > 0.0 else None
                    )
                    # Extract only the new generated text
                    generated = out[0]["generated_text"]
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    return generated
            except Exception as e:
                print(f"✗ HF generation failed: {e}")
                print("Falling back to mock answer")
                return self._mock_answer(question, contexts)
        
        # Use mock answer if no HF model
        return self._mock_answer(question, contexts)

    def _mock_answer(self, question: str, contexts: List[dict]) -> str:
        """Generate a mock answer when no LLM is available."""
        lines = [f"**Question:** {question}\n"]
        lines.append("**Evidence Retrieved:**")
        
        if contexts:
            for i, c in enumerate(contexts[:5]):
                if not c:
                    continue
                source = c.get('source', 'Unknown')
                excerpt = c.get('excerpt', c.get('text', ''))[:200].replace('\n', ' ')
                lines.append(f"[{i+1}] **{source}** — {excerpt}...")
        else:
            lines.append("No relevant documents found.")
        
        lines.append("\n**Suggested Analysis Steps:**")
        lines.append("1. **Quality Control**: Filter cells and genes based on QC metrics")
        lines.append("2. **Normalization**: Apply log-normalization or SCTransform")
        lines.append("3. **Feature Selection**: Identify highly variable genes (HVGs)")
        lines.append("4. **Dimensionality Reduction**: PCA followed by UMAP/t-SNE")
        lines.append("5. **Clustering**: Leiden or Louvain clustering")
        lines.append("6. **Cell Type Annotation**: Use markers or reference-based methods")
        
        lines.append("\n**Common Tools:**")
        lines.append("- **Scanpy**: Python toolkit for analyzing single-cell data")
        lines.append("- **Seurat**: R package for scRNA-seq analysis")
        lines.append("- **scVI**: Deep learning models for scRNA-seq")
        
        lines.append("\n**Model Examples (see src/models/):**")
        lines.append("- Simple classifier: `src/models/supervised.py`")
        lines.append("- GAN for latent vectors: `src/models/gan_latent.py`")
        
        lines.append("\n*Note: This is a mock response. Set HF_MODEL environment variable to use a real language model.*")
        
        return "\n".join(lines)


class RAGChain:
    def __init__(self, embedder: Embedder = None, faiss_store: FaissStore = None):
        self.embedder = embedder
        self.faiss = faiss_store
        self.llm = LLMWrapper()

    def answer(self, question: str, k: int = 5) -> Tuple[str, List[dict], List[float]]:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (answer, retrieved_docs, distances)
        """
        if self.embedder is None or self.faiss is None:
            raise RuntimeError("Embedder and FaissStore must be initialized. Load the index first.")
        
        # Embed the question
        qvec = self.embedder.embed_texts([question])[0]
        
        # Retrieve relevant documents
        hits, dists = self.faiss.search(qvec, k=k)
        
        # Generate answer
        answer = self.llm.generate(question, hits)
        
        return answer, hits, dists

    def rebuild_index_from_papers(
        self,
        papers_folder: str = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        batch_size: int = 16
    ) -> int:
        """
        Rebuild the FAISS index from papers in the specified folder.
        
        Args:
            papers_folder: Path to folder containing papers
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for embedding
            
        Returns:
            Number of chunks indexed
        """
        if papers_folder is None:
            papers_folder = os.path.join("scRAG", "data", "papers")
        
        print(f"Loading documents from {papers_folder}...")
        docs = load_documents_from_papers_folder(papers_folder)
        
        if not docs:
            print("No documents found!")
            return 0
        
        print(f"Found {len(docs)} documents. Chunking...")
        chunks = docs_to_chunks(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Created {len(chunks)} chunks.")
        
        texts = [c["text"] for c in chunks]
        
        # Initialize embedder if needed
        if self.embedder is None:
            print("Initializing embedder...")
            self.embedder = Embedder()
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_texts(texts, batch_size=batch_size)
        dim = embeddings.shape[1]
        print(f"Generated embeddings with dimension {dim}")
        
        # Initialize or reset FAISS store
        if self.faiss is None:
            print("Initializing FAISS store...")
            self.faiss = FaissStore(dim=dim)
        else:
            print("Resetting existing FAISS store...")
            self.faiss.reset()
            self.faiss._init_index(dim)
        
        # Prepare metadata
        metas = [
            {
                "id": c["id"],
                "source": c["source"],
                "source_path": c.get("source_path"),
                "excerpt": c.get("excerpt"),
                "text": c.get("text")
            }
            for c in chunks
        ]
        
        # Add to FAISS
        print("Adding embeddings to FAISS index...")
        self.faiss.add(embeddings, metas)
        
        # Save index
        print("Saving index...")
        self.faiss.save()
        print(f"✓ Index saved successfully with {len(chunks)} chunks!")
        
        return len(chunks)
