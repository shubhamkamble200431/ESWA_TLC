"""
predict.py — Load best checkpoint, run greedy inference on the test set,
             compute overall + bin-wise BLEU/chrF/TER, and save predictions.

Usage:
    python predict.py --config transformer_6
    python predict.py --all          # evaluate all trained models
"""

import argparse
import os
import logging

import torch

from tqdm import tqdm
from config import (
    ALL_CONFIGS, CKPT_DIR, PRED_DIR, OUTPUT_DIR,
    DEVICE, SEED, BIN_NAMES,
)
from dataset import train_bpe, load_tokenizer, get_test_dataset
from torch.utils.data import DataLoader
from dataset import collate_fn
from model import build_model
from utils import (
    set_seed, setup_logging,
    corpus_bleu, corpus_chrf, corpus_ter,
    bin_wise_metrics, save_results,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# Custom test data paths
# ──────────────────────────────────────────────────────

TEST_SRC = "/media/kpdubey/8.0 TB Volume/Shubham/LCS/BASELINE_65k/65k_data/test_7264_5.sa"
TEST_TGT = "/media/kpdubey/8.0 TB Volume/Shubham/LCS/BASELINE_65k/65k_data/test_7264_5.hi"


# ──────────────────────────────────────────────────────
# Token-id → string decoder
# ──────────────────────────────────────────────────────

def decode_batch(ids_tensor: torch.Tensor, sp, bos_id, eos_id, pad_id) -> list:
    texts = []
    for row in ids_tensor.cpu().tolist():
        toks = [t for t in row if t not in (bos_id, pad_id)]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        texts.append(sp.decode(toks))
    return texts


# ──────────────────────────────────────────────────────
# Evaluation for a single config
# ──────────────────────────────────────────────────────

def evaluate_model(config_name: str, batch_size: int = 32):
    set_seed(SEED)
    logger_local = setup_logging(f"predict_{config_name}")
    logger_local.info("=" * 60)
    logger_local.info("Evaluating: %s", config_name)

    cfg    = ALL_CONFIGS[config_name]
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # ── tokeniser ──────────────────────────────────
    train_bpe()
    sp     = load_tokenizer()
    vocab  = sp.get_piece_size()
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # ── checkpoint ─────────────────────────────────
    ckpt_path = os.path.join(CKPT_DIR, f"{config_name}_best.pt")
    if not os.path.exists(ckpt_path):
        logger_local.error("Checkpoint not found: %s  (train first)", ckpt_path)
        return None

    model = build_model(cfg, vocab, pad_id).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger_local.info("Loaded checkpoint from epoch %d (val BLEU %.2f)",
                      ckpt.get("epoch", -1), ckpt.get("best_bleu", 0.0))

    # ── test data ──────────────────────────────────
    test_ds = get_test_dataset(sp, cfg.max_len, src_file=TEST_SRC, tgt_file=TEST_TGT)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ── inference ──────────────────────────────────
    all_hyps    = []
    all_refs    = []
    all_lengths = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"  Inference [{config_name}]",
                    unit="batch", dynamic_ncols=True)
        for src, tgt, lengths in pbar:
            src = src.to(device)
            pred_ids = model.greedy_decode(src, bos_id, eos_id, max_len=cfg.max_len)
            hyps = decode_batch(pred_ids, sp, bos_id, eos_id, pad_id)
            refs = decode_batch(tgt,      sp, bos_id, eos_id, pad_id)
            all_hyps    += hyps
            all_refs    += refs
            all_lengths += lengths.tolist()
            pbar.set_postfix(decoded=len(all_hyps))

    # ── overall metrics ────────────────────────────
    overall = {
        "bleu":  round(corpus_bleu(all_hyps, all_refs),  2),
        "chrf":  round(corpus_chrf(all_hyps, all_refs),  2),
        "ter":   round(corpus_ter(all_hyps,  all_refs),  2),
        "count": len(all_hyps),
    }
    logger_local.info("Overall  BLEU=%.2f  chrF=%.2f  TER=%.2f  (#%d)",
                      overall["bleu"], overall["chrf"], overall["ter"], overall["count"])

    # ── bin-wise metrics ───────────────────────────
    bins = bin_wise_metrics(all_hyps, all_refs, all_lengths)
    for bname in BIN_NAMES:
        b = bins[bname]
        logger_local.info(
            "  Bin %-6s  BLEU=%5.2f  chrF=%5.2f  TER=%5.2f  (#%d)",
            bname, b["bleu"], b["chrf"], b["ter"], b["count"]
        )

    # ── save predictions ───────────────────────────
    pred_file = os.path.join(PRED_DIR, f"{config_name}_predictions.tsv")
    with open(pred_file, "w", encoding="utf-8") as fout:
        fout.write("src_len\treference\thypothesis\n")
        for length, ref, hyp in zip(all_lengths, all_refs, all_hyps):
            fout.write(f"{length}\t{ref}\t{hyp}\n")
    logger_local.info("Predictions saved → %s", pred_file)

    # ── save metrics JSON ──────────────────────────
    metrics = {"overall": overall, "bins": bins}
    metrics_file = os.path.join(OUTPUT_DIR, "results", f"{config_name}_test_metrics.json")
    save_results(metrics, metrics_file)
    logger_local.info("Metrics saved → %s", metrics_file)

    return metrics


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Sanskrit→Hindi NMT model on test set")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", choices=list(ALL_CONFIGS.keys()))
    g.add_argument("--all",    action="store_true", help="Evaluate all trained models")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    to_run  = list(ALL_CONFIGS.keys()) if args.all else [args.config]
    all_res = {}

    for cfg_name in to_run:
        result = evaluate_model(cfg_name, args.batch_size)
        if result:
            all_res[cfg_name] = result

    # ── summary table ──────────────────────────────
    if all_res:
        print("\n" + "="*80)
        print(f"  {'Model':<22} {'BLEU':>6} {'chrF':>6} {'TER':>6}  | Bin-wise BLEU →")
        print("  " + "-"*76)
        header_bins = "  ".join(f"{b:>6}" for b in BIN_NAMES)
        print(f"  {'':22} {'':>6} {'':>6} {'':>6}  | {header_bins}")
        print("="*80)
        for name, res in all_res.items():
            ov = res["overall"]
            bv = "  ".join(f"{res['bins'][b]['bleu']:>6.2f}" for b in BIN_NAMES)
            print(f"  {name:<22} {ov['bleu']:>6.2f} {ov['chrf']:>6.2f} {ov['ter']:>6.2f}  | {bv}")
        print("="*80 + "\n")