#!/usr/bin/env python3
"""
Helper functions for the election model.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def softmax_sample(logits: np.ndarray, rng: np.random.Generator, tau: float) -> int:
    """Sample from a softmax distribution with temperature."""
    z = logits / max(tau, 1e-9)
    z = z - np.max(z)
    probs = np.exp(z)
    probs /= probs.sum()
    return int(rng.choice(len(logits), p=probs))


def allocate_seats_largest_remainder(
    party_votes: pd.Series,
    n_seats: int,
) -> Tuple[pd.Series, float, pd.Series]:
    """Allocate seats using the largest remainder method (Hare quota)."""
    total_valid = float(party_votes.sum())
    eq = total_valid / n_seats if n_seats > 0 else 0.0
    
    if eq <= 0:
        seats = pd.Series(0, index=party_votes.index, dtype=int)
        remainders = pd.Series(0.0, index=party_votes.index)
        return seats, eq, remainders

    qp = np.floor(party_votes / eq).astype(int)
    seats_assigned = int(qp.sum())
    seats = qp.copy()
    remainders = party_votes / eq - qp

    remaining = n_seats - seats_assigned
    if remaining > 0:
        order = remainders.sort_values(ascending=False).index.tolist()
        for idx in order[:remaining]:
            seats.loc[idx] += 1
    elif remaining < 0:
        order = remainders.sort_values(ascending=True).index.tolist()
        for idx in order[: abs(remaining)]:
            seats.loc[idx] = max(0, seats.loc[idx] - 1)

    return seats.astype(int), eq, remainders


def save_plot(x, y, xlabel, ylabel, title, path: Path) -> None:
    """Save a simple line plot to file."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
