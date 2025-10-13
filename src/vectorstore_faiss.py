"""
scRAG/src/vectorstore_faiss.py

Simple FAISS wrapper that saves index + metadata.
"""
import faiss
import numpy as np
import pickle
import os

class FaissStore:
    def __init__(self, dim: int = None, index_path: str = "scRAG/data/faiss_index.index",
                 meta_path: str = "scRAG/data/faiss_meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadatas = []
        if dim is not None:
            self._init_index(dim)

    def _init_index(self, dim: int):
        self.dim = int(dim)
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadatas = []

    def add(self, vectors, metadatas):
        vectors = np.asarray(vectors).astype("float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n, d = vectors.shape
        if self.index is None:
            self._init_index(d)
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, qvec, k: int = 5):
        q = np.asarray(qvec).astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if self.index is None:
            raise RuntimeError("Index not initialized.")
        dists, ids = self.index.search(q, k)
        hits = []
        for idx in ids[0]:
            hits.append(self.metadatas[idx] if idx < len(self.metadatas) else None)
        return hits, dists[0].tolist()

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        if self.index is None:
            raise RuntimeError("No index to save.")
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadatas = pickle.load(f)
            self.dim = int(self.index.d)
            return True
        return False

    def reset(self):
        self.index = None
        self.metadatas = []
        self.dim = None
