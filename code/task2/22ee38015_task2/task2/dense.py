"""Normalized MiniLM embeddings + exact FAISS inner-product (= cosine) search."""
from pathlib import Path
import numpy as np
from common.io import digest, read_json, write_json


class DenseRetriever:
    def __init__(self, corpus, cfg, cache_dir, corpus_hash):
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
        self.corpus = corpus
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(cfg["encoder"], revision=cfg["encoder_revision"], device=device)
        identity = {"corpus_hash": corpus_hash, "model": cfg["encoder"],
                    "revision": cfg["encoder_revision"], "normalized": True}
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vectors_path, metadata_path = cache_dir / "embeddings.npy", cache_dir / "embeddings.meta.json"
        if vectors_path.exists() and metadata_path.exists() and read_json(metadata_path) == identity:
            vectors = np.load(vectors_path)
        else:
            vectors = self.encoder.encode([p["text"] for p in corpus], batch_size=64,
                                          normalize_embeddings=True, show_progress_bar=True,
                                          convert_to_numpy=True).astype("float32")
            np.save(vectors_path, vectors)
            write_json(metadata_path, identity)
        if len(vectors) != len(corpus):
            raise ValueError("Embedding cache does not match the corpus")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.identity = digest(identity)

    def search(self, question, depth=50):
        vector = self.encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(np.ascontiguousarray(vector, dtype="float32"), min(depth, len(self.corpus)))
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
