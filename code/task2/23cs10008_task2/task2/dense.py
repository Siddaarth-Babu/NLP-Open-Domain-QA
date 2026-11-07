"""Dense retrieval helpers for Task 2.

The index uses normalized MiniLM embeddings and FAISS IndexFlatIP.  Inner
product between unit vectors is cosine similarity, which makes the retrieval
rule explicit and reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def embedding_device(requested: str) -> str:
    """Resolve the explicit CPU/GPU choice without silently requiring CUDA."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(model_name: str, device: str, revision: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=embedding_device(device), revision=revision)


def encode_texts(encoder, texts: list[str], batch_size: int) -> np.ndarray:
    """Return float32 L2-normalized vectors suitable for cosine FAISS search."""
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def save_index(vectors: np.ndarray, document_ids: list[str], index_path: Path) -> None:
    import faiss

    if vectors.ndim != 2 or len(vectors) != len(document_ids):
        raise ValueError("one 2-D embedding is required for every document id")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    index_path.with_suffix(index_path.suffix + ".ids.json").write_text(
        json.dumps(document_ids), encoding="utf-8"
    )


def load_index(index_path: Path):
    import faiss

    index = faiss.read_index(str(index_path))
    document_ids = json.loads(index_path.with_suffix(index_path.suffix + ".ids.json").read_text(encoding="utf-8"))
    if index.ntotal != len(document_ids):
        raise ValueError("FAISS index and its document-id sidecar disagree")
    return index, document_ids
