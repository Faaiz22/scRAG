"""
scRAG/src/embeddings.py

Wrapper around sentence-transformers for embeddings.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts, batch_size: int = 32):
        embs = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size)
        return embs.astype("float32")

    def embedding_dim(self):
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            return None
