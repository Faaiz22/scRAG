"""
scRAG/src/rag_chain.py

RAG orchestration: rebuild index helper, retrieval, and LLM wrapper.
Supports optional local HF model via HF_MODEL env var; otherwise uses mock.
"""
import os
from typing import List
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
            try:
                device = 0 if (self.hf_device == "cuda" or (self.hf_device is None and torch.cuda.is_available())) else -1
                try:
                    self.pipeline = pipeline("text2text-generation", model=self.hf_model_name, device=device)
                    self.is_seq2seq = True
                except Exception:
                    self.pipeline = pipeline("text-generation", model=self.hf_model_name, device=device)
                    self.is_seq2seq = False
            except Exception as e:
                print(f"[LLMWrapper] failed to load HF model {self.hf_model_name}: {e}")
                self.pipeline = None

    def generate(self, question: str, contexts: List[dict]) -> str:
        ctx_lines = []
        for i, c in enumerate(contexts[:8]):
            if not c:
                continue
            title = c.get("source", "unknown")
            excerpt = (c.get("excerpt") or c.get("text","")[:400]).replace("\n"," ")
            ctx_lines.append(f"[{i+1}] {title} -- {excerpt[:400]}")
        context_block = "\n".join(ctx_lines)
        prompt = (
            "You are a concise single-cell genomics assistant. Use the retrieved evidence below to answer the user's question.\n\n"
            "Retrieved evidence:\n"
            f"{context_block}\n\n"
            "Instructions: 1) Answer succinctly. 2) Provide numbered reproducible steps if methods asked. 3) When recommending models, include architecture + hyperparameters. Include citations [n].\n\n"
            f"User question: {question}\n\nAnswer:\n"
        )
        if self.pipeline is not None:
            try:
                if self.is_seq2seq:
                    out = self.pipeline(prompt, max_length=self.max_tokens, do_sample=(self.temperature>0.0), temperature=self.temperature)
                    return out[0]["generated_text"]
                else:
                    out = self.pipeline(prompt, max_new_tokens=self.max_tokens, do_sample=(self.temperature>0.0), temperature=self.temperature)
                    return out[0]["generated_text"]
            except Exception as e:
                print("[LLMWrapper] HF generation failed:", e)
                return self._mock_answer(question, contexts)
        return self._mock_answer(question, contexts)

    def _mock_answer(self, question, contexts):
        lines = [f"Question: {question}\n"]
        lines.append("Top evidence (mock):")
        for i, c in enumerate(contexts[:6]):
            if not c: continue
            lines.append(f"[{i+1}] {c.get('source')} — {c.get('excerpt')[:200].replace(chr(10),' ')}")
        lines.append("\nMock steps:")
        lines.append("1) Inspect cited references.")
        lines.append("2) Example pipeline: Scanpy -> Normalize -> HVG -> PCA -> clustering.")
        lines.append("3) For models: HVG->PCA->VAE (latent=16) or small MLP classifier. See src/models/")
        return "\n".join(lines)

class RAGChain:
    def __init__(self, embedder: Embedder = None, faiss_store: FaissStore = None):
        self.embedder = embedder
        self.faiss = faiss_store
        self.llm = LLMWrapper()

    def answer(self, question: str, k: int = 5):
        if self.embedder is None or self.faiss is None:
            raise RuntimeError("Embedder and FaissStore required.")
        qvec = self.embedder.embed_texts([question])[0]
        hits, dists = self.faiss.search(qvec, k=k)
        answer = self.llm.generate(question, hits)
        return answer, hits, dists

    def rebuild_index_from_papers(self, papers_folder: str = None, chunk_size: int = 1200, chunk_overlap: int = 200, batch_size: int = 16):
        if papers_folder is None:
            papers_folder = os.path.join("scRAG", "data", "papers")
        docs = load_documents_from_papers_folder(papers_folder)
        if not docs:
            return 0
        chunks = docs_to_chunks(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = [c["text"] for c in chunks]
        if self.embedder is None:
            self.embedder = Embedder()
        embeddings = self.embedder.embed_texts(texts, batch_size=batch_size)
        dim = embeddings.shape[1]
        if self.faiss is None:
            self.faiss = FaissStore(dim=dim)
        else:
            self.faiss.reset()
            self.faiss._init_index(dim)
        metas = [{"id": c["id"], "source": c["source"], "source_path": c.get("source_path"), "excerpt": c.get("excerpt"), "text": c.get("text")} for c in chunks]
        self.faiss.add(embeddings, metas)
        self.faiss.save()
        return len(chunks)
