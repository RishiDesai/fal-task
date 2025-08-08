from typing import Tuple, Optional

import numpy as np
from sklearn.datasets import make_moons


def make_two_moons(n_samples: int = 10000, noise: float = 0.05, seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    # Standardize to roughly unit variance and zero mean for stable training
    X = X.astype(np.float32)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    return X, y
