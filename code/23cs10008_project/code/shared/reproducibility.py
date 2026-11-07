"""Reproducibility controls shared by neural Task 2 and Task 3 runners."""

from __future__ import annotations

import random


def set_inference_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before deterministic greedy inference."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
