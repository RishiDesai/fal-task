import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons


def to_device() -> torch.device:
    # Prefer MPS on Apple Silicon if available, else CUDA, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def plot_generated_vs_target(gen: np.ndarray, target: np.ndarray, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(target[:, 0], target[:, 1], s=4, alpha=0.6, c="#1f77b4")
    axes[0].set_title("Two Moons (target)")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(gen[:, 0], gen[:, 1], s=4, alpha=0.6, c="#ff7f0e")
    axes[1].set_title("Generated points")
    axes[1].set_aspect("equal", adjustable="box")

    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_generated_only(gen: np.ndarray, save_path: str, title: str = "Generated Points"):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(gen[:, 0], gen[:, 1], s=4, alpha=0.6, c="#ff7f0e")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def make_two_moons(n_samples: int = 10000, noise: float = 0.05, seed: Optional[int] = 42) -> Tuple[
    np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    # Standardize to roughly unit variance and zero mean for stable training
    X = X.astype(np.float32)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    return X, y
