"""
utils.py — Loss functions, learning-rate scheduler, metric helpers,
           bin-wise evaluation utilities, and logging setup.
"""

import os
import math
import json
import logging
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sacrebleu.metrics import BLEU, CHRF, TER

from config import BINS, BIN_NAMES, CLIP_GRAD, LOG_DIR


# ──────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────

def setup_logging(name: str) -> logging.Logger:
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)


# ──────────────────────────────────────────────────────
# Label-smoothed cross-entropy
# ──────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id     = pad_id
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, T, V)
        target : (B, T)
        """
        B, T, V = logits.shape
        logits  = logits.reshape(B * T, V)
        target  = target.reshape(B * T)

        log_prob = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_prob.mean(dim=-1)
        nll_loss    = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        loss        = self.confidence * nll_loss + self.smoothing * smooth_loss

        mask = target != self.pad_id
        return loss[mask].mean()


# ──────────────────────────────────────────────────────
# Noam / Warmup-then-decay scheduler
# ──────────────────────────────────────────────────────

class NoamScheduler:
    """
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self._step        = 0
        self._rate        = 0.0

    def step(self):
        self._step += 1
        rate = self._compute_lr()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _compute_lr(self) -> float:
        s = self._step
        return (
            self.d_model ** (-0.5)
            * min(s ** (-0.5), s * self.warmup_steps ** (-1.5))
        )

    @property
    def last_lr(self) -> float:
        return self._rate


# ──────────────────────────────────────────────────────
# Sacrebleu wrapper — sentence + corpus
# ──────────────────────────────────────────────────────

_bleu_scorer = BLEU(effective_order=True)
_chrf_scorer = CHRF()
_ter_scorer  = TER()


def corpus_bleu(hypotheses: List[str], references: List[str]) -> float:
    result = _bleu_scorer.corpus_score(hypotheses, [references])
    return result.score


def corpus_chrf(hypotheses: List[str], references: List[str]) -> float:
    result = _chrf_scorer.corpus_score(hypotheses, [references])
    return result.score


def corpus_ter(hypotheses: List[str], references: List[str]) -> float:
    result = _ter_scorer.corpus_score(hypotheses, [references])
    return result.score


# ──────────────────────────────────────────────────────
# Bin assignment
# ──────────────────────────────────────────────────────

def assign_bin(length: int) -> int:
    """Return 0-indexed bin index for a source length."""
    for i, (lo, hi) in enumerate(BINS):
        if hi is None:
            return i
        if lo <= length <= hi:
            return i
    return len(BINS) - 1   # fallback


def bin_wise_metrics(
    hypotheses: List[str],
    references: List[str],
    src_lengths: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    Compute BLEU, chrF, TER per length bin.
    Returns {bin_name: {"bleu": ..., "chrf": ..., "ter": ..., "count": ...}}
    """
    buckets: Dict[int, Tuple[List[str], List[str]]] = {i: ([], []) for i in range(len(BINS))}
    for h, r, l in zip(hypotheses, references, src_lengths):
        b = assign_bin(l)
        buckets[b][0].append(h)
        buckets[b][1].append(r)

    results = {}
    for i, name in enumerate(BIN_NAMES):
        hyps, refs = buckets[i]
        if not hyps:
            results[name] = {"bleu": 0.0, "chrf": 0.0, "ter": 0.0, "count": 0}
            continue
        results[name] = {
            "bleu":  round(corpus_bleu(hyps, refs),  2),
            "chrf":  round(corpus_chrf(hyps, refs),  2),
            "ter":   round(corpus_ter(hyps, refs),   2),
            "count": len(hyps),
        }
    return results


# ──────────────────────────────────────────────────────
# Gradient clipping helper (wrap into training loop)
# ──────────────────────────────────────────────────────

def clip_and_step(model: nn.Module, scheduler, clip: float = CLIP_GRAD):
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scheduler.step()


# ──────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_bleu", 0.0)


# ──────────────────────────────────────────────────────
# JSON results I/O
# ──────────────────────────────────────────────────────

def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)