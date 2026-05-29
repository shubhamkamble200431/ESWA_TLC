"""
visualize.py — Generate all plots and visualizations.

Plots produced per model:
  1. Training curve          (train loss / val loss / val BLEU)
  2. Bin-wise BLEU bar chart
  3. Bin-wise chrF bar chart
  4. Bin-wise TER  bar chart

Aggregate plots (all models together):
  5. Overall BLEU comparison
  6. Bin-wise BLEU heatmap
  7. Radar chart — per-model bin BLEU
  8. Sentence-length distribution of test set (from train.sa lengths)

Usage:
    python visualize.py                 # reads all results and plots
    python visualize.py --config transformer_6
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

from config import (
    ALL_CONFIGS, BIN_NAMES, PLOT_DIR, OUTPUT_DIR, DATA_DIR,
    TRANSFORMER_CONFIGS, MAMBA_CONFIGS,
)

# ── colour palette ────────────────────────────────────
TRANSFORMER_COLORS = ["#2E86AB", "#A23B72", "#F18F01"]   # blue, purple, amber
MAMBA_COLORS       = ["#C73E1D", "#3B1F2B", "#44BBA4"]   # red, dark, teal
ALL_COLORS         = TRANSFORMER_COLORS + MAMBA_COLORS

MODEL_ORDER = list(ALL_CONFIGS.keys())


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def _results_path(config_name: str) -> str:
    return os.path.join(OUTPUT_DIR, "results", f"{config_name}_test_metrics.json")


def _history_path(config_name: str) -> str:
    return os.path.join(OUTPUT_DIR, "results", f"{config_name}_train_history.json")


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, name: str):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────
# 1. Training curves  (per model)
# ─────────────────────────────────────────────────────

def plot_training_curve(config_name: str):
    hist = _load_json(_history_path(config_name))
    if hist is None:
        return

    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss", color="#2E86AB", linewidth=2)
    axes[0].plot(epochs, hist["val_loss"],   label="Val Loss",   color="#C73E1D", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{config_name} — Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # BLEU
    axes[1].plot(epochs, hist["val_bleu"], label="Val BLEU", color="#44BBA4", linewidth=2, marker="o", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BLEU")
    axes[1].set_title(f"{config_name} — Validation BLEU")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Training History: {config_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"{config_name}_training_curve.png")


# ─────────────────────────────────────────────────────
# 2–4. Per-model bin-wise bar charts
# ─────────────────────────────────────────────────────

def plot_binwise_bars(config_name: str):
    res = _load_json(_results_path(config_name))
    if res is None:
        return

    bins_data = res["bins"]
    x         = np.arange(len(BIN_NAMES))
    metrics   = ["bleu", "chrf", "ter"]
    titles    = ["BLEU", "chrF", "TER"]
    colors    = ["#2E86AB", "#44BBA4", "#C73E1D"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title, col in zip(axes, metrics, titles, colors):
        vals   = [bins_data[b][metric] for b in BIN_NAMES]
        counts = [bins_data[b]["count"] for b in BIN_NAMES]
        bars   = ax.bar(x, vals, color=col, alpha=0.85, edgecolor="white", linewidth=0.8)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"n={cnt}", ha="center", va="bottom", fontsize=7, color="#333")
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_NAMES)
        ax.set_xlabel("Source Length Bin")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by Length Bin")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Bin-wise Metrics: {config_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"{config_name}_binwise_metrics.png")


# ─────────────────────────────────────────────────────
# 5. Overall BLEU comparison (all models)
# ─────────────────────────────────────────────────────

def plot_overall_comparison(all_results: dict):
    names  = [n for n in MODEL_ORDER if n in all_results]
    bleus  = [all_results[n]["overall"]["bleu"]  for n in names]
    chrfs  = [all_results[n]["overall"]["chrf"]  for n in names]
    ters   = [all_results[n]["overall"]["ter"]   for n in names]
    colors = [ALL_COLORS[MODEL_ORDER.index(n)] for n in names]

    x   = np.arange(len(names))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))

    bars1 = ax.bar(x - w,   bleus, w, label="BLEU",  color="#2E86AB", alpha=0.9)
    bars2 = ax.bar(x,       chrfs, w, label="chrF",  color="#44BBA4", alpha=0.9)
    bars3 = ax.bar(x + w,   ters,  w, label="TER ↓", color="#C73E1D", alpha=0.9)

    for bar in list(bars1) + list(bars2) + list(bars3):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Overall Test Metrics: All Models", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "all_models_overall_comparison.png")


# ─────────────────────────────────────────────────────
# 6. Bin-wise BLEU heatmap
# ─────────────────────────────────────────────────────

def plot_binwise_heatmap(all_results: dict):
    names  = [n for n in MODEL_ORDER if n in all_results]
    matrix = np.array([
        [all_results[n]["bins"][b]["bleu"] for b in BIN_NAMES]
        for n in names
    ])

    fig, ax = plt.subplots(figsize=(10, len(names) * 0.7 + 1.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="BLEU")

    ax.set_xticks(range(len(BIN_NAMES)))
    ax.set_xticklabels(BIN_NAMES)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Source Length Bin")
    ax.set_title("Bin-wise BLEU Heatmap", fontsize=13, fontweight="bold")

    for i in range(len(names)):
        for j in range(len(BIN_NAMES)):
            ax.text(j, i, f"{matrix[i, j]:.1f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if matrix[i, j] < matrix.max() * 0.7 else "white")

    plt.tight_layout()
    _save(fig, "all_models_binwise_heatmap.png")


# ─────────────────────────────────────────────────────
# 7. Radar chart — per-model bin BLEU
# ─────────────────────────────────────────────────────

def plot_radar(all_results: dict):
    names  = [n for n in MODEL_ORDER if n in all_results]
    cats   = BIN_NAMES
    N      = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, name in enumerate(names):
        vals = [all_results[name]["bins"][b]["bleu"] for b in cats]
        vals += vals[:1]
        color = ALL_COLORS[idx % len(ALL_COLORS)]
        ax.plot(angles, vals, linewidth=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_title("Bin-wise BLEU Radar Chart", y=1.08, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    _save(fig, "all_models_radar.png")


# ─────────────────────────────────────────────────────
# 8. Source sentence length distribution
# ─────────────────────────────────────────────────────

def plot_length_distribution():
    train_sa = os.path.join(DATA_DIR, "train.sa")
    if not os.path.exists(train_sa):
        print("  Skipping length distribution — train.sa not found.")
        return

    with open(train_sa, encoding="utf-8") as f:
        lengths = [len(line.split()) for line in f]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # histogram
    axes[0].hist(lengths, bins=40, color="#2E86AB", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Source Length (words)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Training Set: Source Length Distribution")
    axes[0].grid(axis="y", alpha=0.3)

    # bin counts
    from config import BINS
    bin_counts = []
    for lo, hi in BINS:
        if hi is None:
            bin_counts.append(sum(1 for l in lengths if l >= lo))
        else:
            bin_counts.append(sum(1 for l in lengths if lo <= l <= hi))

    axes[1].bar(BIN_NAMES, bin_counts, color="#44BBA4", edgecolor="white", alpha=0.85)
    for i, cnt in enumerate(bin_counts):
        axes[1].text(i, cnt + max(bin_counts) * 0.01, str(cnt),
                     ha="center", fontsize=9)
    axes[1].set_xlabel("Length Bin")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Training Set: Sentence Counts per Bin")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Sanskrit Source Sentence Lengths (train.sa)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "train_length_distribution.png")


# ─────────────────────────────────────────────────────
# 9. Grouped bin-wise BLEU comparison (transformers vs mamba)
# ─────────────────────────────────────────────────────

def plot_group_binwise(all_results: dict):
    """Two side-by-side panels: Transformers | Mamba."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    groups = [
        ("Transformer Variants", list(TRANSFORMER_CONFIGS.keys()), TRANSFORMER_COLORS),
        ("Mamba Variants",       list(MAMBA_CONFIGS.keys()),       MAMBA_COLORS),
    ]

    x = np.arange(len(BIN_NAMES))
    for ax, (title, names, cols) in zip(axes, groups):
        available = [n for n in names if n in all_results]
        w = 0.8 / max(len(available), 1)
        for i, (name, col) in enumerate(zip(available, cols)):
            vals = [all_results[name]["bins"][b]["bleu"] for b in BIN_NAMES]
            offset = (i - len(available) / 2 + 0.5) * w
            ax.bar(x + offset, vals, w * 0.9, label=name, color=col, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_NAMES)
        ax.set_xlabel("Length Bin")
        ax.set_ylabel("BLEU")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Bin-wise BLEU: Transformers vs Mamba", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "all_models_group_binwise_bleu.png")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main(config_name=None):
    os.makedirs(PLOT_DIR, exist_ok=True)

    # load all available results
    all_results = {}
    for name in MODEL_ORDER:
        r = _load_json(_results_path(name))
        if r is not None:
            all_results[name] = r

    to_plot = [config_name] if config_name else MODEL_ORDER

    print("\n── Per-model plots ──────────────────────────────")
    for name in to_plot:
        if _load_json(_history_path(name)):
            plot_training_curve(name)
        if name in all_results:
            plot_binwise_bars(name)

    if len(all_results) > 1:
        print("\n── Aggregate plots ──────────────────────────────")
        plot_overall_comparison(all_results)
        plot_binwise_heatmap(all_results)
        plot_radar(all_results)
        plot_group_binwise(all_results)

    print("\n── Length distribution ──────────────────────────")
    plot_length_distribution()

    print(f"\nAll plots saved to: {PLOT_DIR}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(ALL_CONFIGS.keys()), default=None,
                   help="Plot only this config (default: all available)")
    args = p.parse_args()
    main(args.config)