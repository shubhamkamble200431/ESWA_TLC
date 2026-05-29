"""
train.py — Training loop for a single model config.
Runs validation every epoch; saves best checkpoint; supports early stopping.

Usage:
    python train.py --config transformer_6
    python train.py --config mamba_base
    python train.py --all           # train every config sequentially
"""

import argparse
import logging
import os
import time

import torch
import torch.optim as optim
from tqdm import tqdm

from config import (
    ALL_CONFIGS, TRANSFORMER_CONFIGS, MAMBA_CONFIGS,
    CKPT_DIR, OUTPUT_DIR, DEVICE, SEED,
    WARMUP_STEPS, LABEL_SMOOTH, PATIENCE,
)
from dataset import train_bpe, load_tokenizer, get_dataloaders
from model import build_model, count_parameters
from utils import (
    set_seed, setup_logging,
    LabelSmoothingLoss, NoamScheduler,
    clip_and_step, save_checkpoint, load_checkpoint,
    corpus_bleu, save_results,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Decode a batch of ids → strings (strip BOS/EOS/PAD)
# ──────────────────────────────────────────────────────

def decode_batch(ids_tensor: torch.Tensor, sp, bos_id: int, eos_id: int, pad_id: int) -> list:
    texts = []
    for row in ids_tensor.cpu().tolist():
        toks = []
        for t in row:
            if t in (bos_id, pad_id):
                continue
            if t == eos_id:
                break
            toks.append(t)
        texts.append(sp.decode(toks))
    return texts


# ──────────────────────────────────────────────────────
# One training epoch
# ──────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, scheduler, device, sp, epoch: int):
    model.train()
    total_loss = 0.0
    steps      = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:>3} [train]", unit="batch",
                leave=False, dynamic_ncols=True)
    for src, tgt, _ in pbar:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        scheduler.zero_grad()
        logits = model(src, tgt_in)
        loss   = criterion(logits, tgt_out)
        loss.backward()
        clip_and_step(model, scheduler)

        total_loss += loss.item()
        steps      += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.last_lr:.2e}")

    return total_loss / max(steps, 1)


# ──────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, sp, bos_id, eos_id, pad_id, epoch: int):
    model.eval()
    total_loss = 0.0
    steps      = 0
    hyps = []
    refs = []

    pbar = tqdm(loader, desc=f"  Epoch {epoch:>3} [val]  ", unit="batch",
                leave=False, dynamic_ncols=True)
    for src, tgt, _ in pbar:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits  = model(src, tgt_in)
        loss    = criterion(logits, tgt_out)
        total_loss += loss.item()
        steps      += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        pred_ids = model.greedy_decode(src, bos_id, eos_id)
        hyps    += decode_batch(pred_ids,  sp, bos_id, eos_id, pad_id)
        refs    += decode_batch(tgt,       sp, bos_id, eos_id, pad_id)

    val_loss = total_loss / max(steps, 1)
    val_bleu = corpus_bleu(hyps, refs)
    return val_loss, val_bleu


# ──────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────

def train_model(config_name: str):
    set_seed(SEED)
    logger_local = setup_logging(f"train_{config_name}")
    logger_local.info("=" * 60)
    logger_local.info("Training config: %s", config_name)

    cfg    = ALL_CONFIGS[config_name]
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    logger_local.info("Device: %s", device)

    # ── tokeniser ──────────────────────────────────
    train_bpe()
    sp     = load_tokenizer()
    vocab  = sp.get_piece_size()
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # ── data ───────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        sp, batch_size=cfg.batch_size, max_len=cfg.max_len
    )

    # ── model ──────────────────────────────────────
    model = build_model(cfg, vocab, pad_id).to(device)
    logger_local.info(
        "Model: %s  |  Parameters: %s",
        config_name, f"{count_parameters(model):,}"
    )

    criterion = LabelSmoothingLoss(vocab, pad_id, LABEL_SMOOTH)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, cfg.d_model, WARMUP_STEPS)

    ckpt_path = os.path.join(CKPT_DIR, f"{config_name}_best.pt")

    # ── training loop ──────────────────────────────
    best_bleu    = 0.0
    patience_ctr = 0
    history      = {"train_loss": [], "val_loss": [], "val_bleu": []}

    epoch_bar = tqdm(range(1, cfg.max_epochs + 1), desc=f"[{config_name}]",
                     unit="epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, criterion, scheduler, device, sp, epoch)
        val_loss, val_bleu = validate(
            model, val_loader, criterion, device, sp, bos_id, eos_id, pad_id, epoch
        )
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bleu"].append(val_bleu)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_bleu=f"{val_bleu:.2f}",
            best=f"{best_bleu:.2f}",
        )

        logger_local.info(
            "Epoch %d/%d  |  train_loss=%.4f  val_loss=%.4f  "
            "val_bleu=%.2f  lr=%.2e  time=%.1fs",
            epoch, cfg.max_epochs,
            train_loss, val_loss, val_bleu,
            scheduler.last_lr, elapsed,
        )

        if val_bleu > best_bleu:
            best_bleu    = val_bleu
            patience_ctr = 0
            save_checkpoint(
                {
                    "epoch":          epoch,
                    "model_state":    model.state_dict(),
                    "optimizer_state":optimizer.state_dict(),
                    "best_bleu":      best_bleu,
                    "config_name":    config_name,
                    "vocab_size":     vocab,
                },
                ckpt_path,
            )
            logger_local.info("  ✓ New best checkpoint saved (BLEU %.2f)", best_bleu)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger_local.info("Early stopping after %d epochs with no improvement.", epoch)
                break

    # save training history
    results_path = os.path.join(OUTPUT_DIR, "results", f"{config_name}_train_history.json")
    save_results(history, results_path)
    logger_local.info("Training done. Best val BLEU: %.2f", best_bleu)
    return best_bleu


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Sanskrit→Hindi NMT model")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config",      choices=list(ALL_CONFIGS.keys()),
                   help="Single config to train")
    g.add_argument("--transformers", action="store_true",
                   help="Train all Transformer configs (6/12/18)")
    g.add_argument("--mambas",       action="store_true",
                   help="Train all Mamba configs")
    g.add_argument("--all",          action="store_true",
                   help="Train all configs sequentially")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    results = {}

    if args.config:
        to_run = [args.config]
    elif args.transformers:
        to_run = list(TRANSFORMER_CONFIGS.keys())
    elif args.mambas:
        to_run = list(MAMBA_CONFIGS.keys())
    else:  # --all
        to_run = list(ALL_CONFIGS.keys())

    print(f"\n{'='*60}")
    print(f"  Running {len(to_run)} config(s): {to_run}")
    print(f"{'='*60}\n")

    for cfg_name in to_run:
        bleu = train_model(cfg_name)
        results[cfg_name] = {"val_bleu": bleu}

    print("\n" + "="*60)
    print("  Summary of best validation BLEU scores")
    print("="*60)
    for name, r in results.items():
        print(f"  {name:25s}  BLEU = {r['val_bleu']:.2f}")
    print("="*60 + "\n")