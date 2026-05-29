import os
import glob
import json
import argparse
import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

BUCKET_LABELS  = ["B1 (1–5)", "B2 (6–10)", "B3 (11–15)", "B4 (16–20)", "B5 (20+)"]
BUCKET_COLORS  = ["#60a5fa", "#34d399", "#a78bfa", "#fb923c", "#f472b6"]
CASE_COLORS    = {
    "C-0": "#94a3b8", "C-1": "#60a5fa", "C-2": "#34d399",
    "C-3": "#a78bfa", "C-4": "#fb923c", "C-5": "#f472b6",
    "C-6": "#facc15", "C-7": "#67e8f9", "C-8": "#86efac",
}

plt.rcParams.update({
    "figure.facecolor":  "#0a1628",
    "axes.facecolor":    "#0f172a",
    "axes.edgecolor":    "#1e293b",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e293b",
    "grid.linewidth":    0.8,
    "legend.facecolor":  "#0f172a",
    "legend.edgecolor":  "#1e293b",
    "legend.labelcolor": "#94a3b8",
    "font.family":       "monospace",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
})


def read_tensorboard(output_dir: str) -> Dict[str, List]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        raise ImportError(
            "tensorboard required for reading logs.\n"
            "Install: pip install tensorboard"
        )

    ea = EventAccumulator(output_dir)
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data


def plot_single_experiment(output_dir: str, save_dir: Optional[str] = None):
    case_id = os.path.basename(output_dir.rstrip("/")).split("_")[0]
    save_dir = save_dir or os.path.join(output_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Reading TensorBoard logs from: {output_dir}")
    try:
        tb = read_tensorboard(output_dir)
    except Exception as e:
        logger.warning(f"TensorBoard read failed: {e}. Attempting JSON fallback.")
        tb = {}

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"Training Dashboard — {case_id}  |  SAN→HIN NMT",
        fontsize=14, fontweight="bold", color="#f8fafc", y=0.98,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _plot_scalar(ax1, tb, "train/loss", "Training Loss", "#60a5fa",
                 xlabel="Step", ylabel="Loss")

    ax2 = fig.add_subplot(gs[0, 1])
    _plot_scalar(ax2, tb, "eval/bleu", "Validation BLEU (Overall)", "#34d399",
                 xlabel="Step", ylabel="BLEU")

    ax4 = fig.add_subplot(gs[1, 0:2])
    _plot_bucket_lines(ax4, tb, "Per-Bucket BLEU over Training")

    ax5 = fig.add_subplot(gs[1, 2])
    _plot_scalar(ax5, tb, "train/learning_rate", "Learning Rate", "#facc15",
                 xlabel="Step", ylabel="LR", log_scale=True)

    _save(fig, os.path.join(save_dir, f"{case_id}_training_dashboard.png"))

    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.suptitle(
        f"Final Per-Bucket BLEU — {case_id}",
        fontsize=13, fontweight="bold", color="#f8fafc",
    )
    _plot_bucket_bars(ax, tb, case_id)
    _save(fig2, os.path.join(save_dir, f"{case_id}_bucket_bleu.png"))

    fig3, ax = plt.subplots(figsize=(10, 5))
    fig3.suptitle(
        f"Length Ratio per Bucket — {case_id}",
        fontsize=13, fontweight="bold", color="#f8fafc",
    )
    _plot_length_ratio(ax, tb, case_id)
    _save(fig3, os.path.join(save_dir, f"{case_id}_length_ratio.png"))

    logger.info(f"Plots saved to: {save_dir}")
    return save_dir


def plot_comparison(output_dirs: List[str], save_dir: str = "outputs/comparison"):
    os.makedirs(save_dir, exist_ok=True)

    all_tb = {}
    for d in output_dirs:
        cid = _infer_case_id(d)
        try:
            all_tb[cid] = read_tensorboard(d)
        except Exception as e:
            logger.warning(f"{cid}: TB read failed ({e})")
            all_tb[cid] = {}

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "Ablation Comparison — SAN→HIN NMT  |  All Cases",
        fontsize=14, fontweight="bold", color="#f8fafc", y=0.99,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Training Loss (All Cases)")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    for cid, tb in all_tb.items():
        data = tb.get("train/loss", [])
        if data:
            steps, vals = zip(*data)
            ax1.plot(steps, _smooth(vals), color=CASE_COLORS.get(cid, "#fff"),
                     label=cid, linewidth=1.8, alpha=0.9)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Final Validation BLEU (Overall)")
    ax2.set_ylabel("BLEU")
    ax2.grid(True, alpha=0.3, axis="y")
    cases, bleus = [], []
    for cid, tb in all_tb.items():
        data = tb.get("eval/bleu", [])
        if data:
            cases.append(cid)
            bleus.append(data[-1][1])
    if cases:
        colors = [CASE_COLORS.get(c, "#94a3b8") for c in cases]
        bars = ax2.bar(cases, bleus, color=colors, edgecolor="#0a1628", linewidth=0.5)
        for bar, val in zip(bars, bleus):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#e2e8f0")

    ax3 = fig.add_subplot(gs[1, :])
    _plot_grouped_bucket_bars(ax3, all_tb)

    _save(fig, os.path.join(save_dir, "comparison_dashboard.png"))

    logger.info(f"Comparison plots saved to: {save_dir}")


def plot_from_json(json_paths: List[str], save_dir: str = "outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}
    for path in json_paths:
        path = path.strip()
        matches = glob.glob(path) if "*" in path else [path]
        for p in matches:
            if not os.path.exists(p):
                continue
            with open(p) as f:
                data = json.load(f)
            cid = data.get("case_id", os.path.basename(p).split("_")[0])
            all_results[cid] = data.get("metrics", {})

    if not all_results:
        logger.error("No valid JSON result files found.")
        return

    bucket_keys = {
        "B1 (1–5)":   "bleu_B1_1-5",
        "B2 (6–10)":  "bleu_B2_6-10",
        "B3 (11–15)": "bleu_B3_11-15",
        "B4 (16–20)": "bleu_B4_16-20",
        "B5 (20+)":   "bleu_B5_20+",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Evaluation Results from JSON — SAN→HIN NMT",
                 fontsize=13, fontweight="bold", color="#f8fafc")

    ax = axes[0]
    ax.set_title("Per-Bucket BLEU by Case")
    ax.set_ylabel("BLEU"); ax.grid(True, alpha=0.3, axis="y")
    cases = sorted(all_results.keys())
    x = np.arange(len(BUCKET_LABELS))
    width = 0.8 / max(len(cases), 1)
    for i, cid in enumerate(cases):
        m = all_results[cid]
        vals = [m.get(v, 0) for v in bucket_keys.values()]
        offset = (i - len(cases)/2 + 0.5) * width
        ax.bar(x + offset, vals, width=width*0.9,
               color=CASE_COLORS.get(cid, "#94a3b8"),
               label=cid, edgecolor="#0a1628", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_LABELS, fontsize=8)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.set_title("Overall BLEU per Case")
    ax2.set_ylabel("BLEU"); ax2.grid(True, alpha=0.3, axis="y")
    case_names = cases
    overall = [all_results[c].get("bleu", 0) for c in cases]
    xp = np.arange(len(cases))
    ax2.bar(xp, overall, 0.6, color="#34d399", label="Overall BLEU", edgecolor="#0a1628")
    ax2.set_xticks(xp); ax2.set_xticklabels(case_names, fontsize=9)
    ax2.legend(fontsize=8)

    _save(fig, os.path.join(save_dir, "eval_from_json.png"))
    logger.info(f"JSON plots saved to: {save_dir}")


def _plot_scalar(ax, tb, tag, title, color, xlabel="Step", ylabel="",
                 hline=None, hline_label=None, hline_color="#ef4444",
                 log_scale=False):
    data = tb.get(tag, [])
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if data:
        steps, vals = zip(*data)
        ax.plot(steps, _smooth(vals), color=color, linewidth=2)
        ax.fill_between(steps, _smooth(vals), alpha=0.1, color=color)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#475569", fontsize=9)
    if hline is not None:
        ax.axhline(hline, color=hline_color, linestyle="--",
                   linewidth=1.2, label=hline_label)
        ax.legend(fontsize=8)
    if log_scale and data:
        ax.set_yscale("log")


def _plot_bucket_lines(ax, tb, title):
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("BLEU")
    ax.grid(True, alpha=0.3)
    bucket_tags = [
        ("eval/bleu_B1_1-5",   "B1 (1–5)"),
        ("eval/bleu_B2_6-10",  "B2 (6–10)"),
        ("eval/bleu_B3_11-15", "B3 (11–15)"),
        ("eval/bleu_B4_16-20", "B4 (16–20)"),
        ("eval/bleu_B5_20+",   "B5 (20+)"),
    ]
    has_data = False
    for (tag, label), color in zip(bucket_tags, BUCKET_COLORS):
        data = tb.get(tag, [])
        if data:
            steps, vals = zip(*data)
            ax.plot(steps, _smooth(vals), color=color, label=label,
                    linewidth=2, marker="o", markersize=2)
            has_data = True
    if has_data:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No bucket data yet\n(appears after first eval)",
                ha="center", va="center", transform=ax.transAxes,
                color="#475569", fontsize=9)


def _plot_bucket_bars(ax, tb, case_id):
    bucket_tags = [
        "eval/bleu_B1_1-5", "eval/bleu_B2_6-10", "eval/bleu_B3_11-15",
        "eval/bleu_B4_16-20", "eval/bleu_B5_20+",
    ]
    vals = []
    for tag in bucket_tags:
        data = tb.get(tag, [])
        vals.append(data[-1][1] if data else 0)

    ax.set_ylabel("BLEU"); ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"Per-Bucket BLEU — {case_id}")
    bars = ax.bar(BUCKET_LABELS, vals, color=BUCKET_COLORS,
                  edgecolor="#0a1628", linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=9, color="#e2e8f0")


def _plot_length_ratio(ax, tb, case_id):
    lr_tags = [
        "eval/length_ratio_B1_1-5", "eval/length_ratio_B2_6-10",
        "eval/length_ratio_B3_11-15", "eval/length_ratio_B4_16-20",
        "eval/length_ratio_B5_20+",
    ]
    vals = []
    for tag in lr_tags:
        data = tb.get(tag, [])
        vals.append(data[-1][1] if data else 0)

    ax.set_ylabel("Length Ratio (pred/ref)"); ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"Length Ratio per Bucket — {case_id}")
    ax.axhline(1.0, color="#facc15", linestyle="--", linewidth=1.5,
               label="Target = 1.0")
    ax.bar(BUCKET_LABELS, vals, color=BUCKET_COLORS,
           edgecolor="#0a1628", linewidth=0.5)
    ax.legend(fontsize=8)
    for i, val in enumerate(vals):
        if val > 0:
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, color="#e2e8f0")


def _plot_grouped_bucket_bars(ax, all_tb: dict):
    ax.set_title("Per-Bucket BLEU — All Cases")
    ax.set_ylabel("BLEU"); ax.grid(True, alpha=0.3, axis="y")
    bucket_tags = [
        "eval/bleu_B1_1-5", "eval/bleu_B2_6-10", "eval/bleu_B3_11-15",
        "eval/bleu_B4_16-20", "eval/bleu_B5_20+",
    ]
    cases = sorted(all_tb.keys())
    x = np.arange(len(BUCKET_LABELS))
    width = 0.8 / max(len(cases), 1)
    for i, cid in enumerate(cases):
        tb = all_tb[cid]
        vals = []
        for tag in bucket_tags:
            data = tb.get(tag, [])
            vals.append(data[-1][1] if data else 0)
        offset = (i - len(cases)/2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=CASE_COLORS.get(cid, "#94a3b8"),
               label=cid, edgecolor="#0a1628", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.legend(fontsize=8, ncol=3)


def _smooth(values, window=5):
    if len(values) < window:
        return list(values)
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    for i in range(window // 2):
        smoothed[i] = arr[:i+1].mean()
        smoothed[-(i+1)] = arr[-(i+1):].mean()
    return smoothed.tolist()


def _infer_case_id(path: str) -> str:
    base = os.path.basename(path.rstrip("/"))
    for c in ["C-0","C-1","C-2","C-3","C-4","C-5","C-6","C-7","C-8"]:
        if c in base:
            return c
    return base[:6]


def _save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Plot training curves and evaluation metrics for SAN→HIN NMT"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Single experiment output dir (contains TensorBoard events)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare multiple cases"
    )
    parser.add_argument(
        "--output_dirs", nargs="+", default=[],
        help="List of output dirs for comparison"
    )
    parser.add_argument(
        "--from_json", nargs="+", default=[],
        help="Plot from JSON eval result files (glob patterns ok)"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Where to save plots (default: <output_dir>/plots)"
    )
    args = parser.parse_args()

    if args.from_json:
        plot_from_json(args.from_json, save_dir=args.save_dir or "outputs/plots")

    elif args.compare:
        dirs = args.output_dirs
        if not dirs and args.output_dir:
            parent = os.path.dirname(args.output_dir.rstrip("/"))
            dirs = sorted(glob.glob(os.path.join(parent, "C-*")))
        plot_comparison(dirs, save_dir=args.save_dir or "outputs/comparison")

    elif args.output_dir:
        plot_single_experiment(args.output_dir, save_dir=args.save_dir)

    else:
        parser.print_help()
