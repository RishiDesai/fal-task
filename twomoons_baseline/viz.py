from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_generated_vs_target(gen: np.ndarray, target: np.ndarray, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(target[:, 0], target[:, 1], s=4, alpha=0.6, c="#1f77b4")
    axes[0].set_title("Two Moons (target)")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(gen[:, 0], gen[:, 1], s=4, alpha=0.6, c="#ff7f0e")
    axes[1].set_title("Generated (baseline)")
    axes[1].set_aspect("equal", adjustable="box")

    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
