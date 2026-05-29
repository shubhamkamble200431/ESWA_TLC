"""
predict_pretrained.py — Inference + bin-wise evaluation for fine-tuned
                        pretrained HuggingFace models (M2M100, LongT5, mT5).

Loads the best checkpoint saved by finetune.py and runs beam-search decoding
on the test set, then computes overall + bin-wise BLEU/chrF/TER.

Usage:
    python predict_pretrained.py --config m2m100_418m
    python predict_pretrained.py --config mt5_base
    python predict_pretrained.py --all
    python predict_pretrained.py --group longt5
"""

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    ALL_PRETRAINED_CONFIGS, PRETRAINED_CONFIGS,
    CKPT_DIR, PRED_DIR, OUTPUT_DIR, DEVICE, SEED, BIN_NAMES,
)
from pretrained_dataset import (
    load_hf_tokenizer, get_pretrained_test_dataset, PretrainedCollator,
)
from utils import (
    set_seed, setup_logging,
    corpus_bleu, corpus_chrf, corpus_ter,
    bin_wise_metrics, save_results,
)
from finetune import GROUP_MAP

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Model loader from saved checkpoint
# ─────────────────────────────────────────────────────────────────

def load_finetuned_model(cfg, device):
    from transformers import (
        M2M100ForConditionalGeneration,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        M2M100Tokenizer,
    )

    ckpt_path = os.path.join(CKPT_DIR, f"{cfg.name}_best")
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python finetune.py --config {cfg.name}"
        )

    if cfg.model_type == "m2m100":
        model     = M2M100ForConditionalGeneration.from_pretrained(ckpt_path)
        tokenizer = M2M100Tokenizer.from_pretrained(ckpt_path)
        tokenizer.src_lang = cfg.src_lang
        tokenizer.tgt_lang = cfg.tgt_lang
    else:
        model     = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    model = model.to(device)
    model.eval()
    logger.info("Loaded fine-tuned checkpoint: %s", ckpt_path)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_pretrained(config_name: str, batch_size: int = 16):
    set_seed(SEED)
    log = setup_logging(f"predict_{config_name}")
    log.info("=" * 60)
    log.info("Evaluating: %s", config_name)

    cfg    = ALL_PRETRAINED_CONFIGS[config_name]
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # ── load model + tokeniser ──────────────
    try:
        model, tokenizer = load_finetuned_model(cfg, device)
    except FileNotFoundError as e:
        log.error(str(e))
        return None

    # ── test data ───────────────────────────
    test_ds   = get_pretrained_test_dataset(cfg)
    collator  = PretrainedCollator(tokenizer, cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # ── generation kwargs ───────────────────
    gen_kwargs = dict(max_new_tokens=cfg.max_tgt_len, num_beams=cfg.num_beams)
    if cfg.model_type == "m2m100":
        tgt_code = cfg.tgt_lang if cfg.tgt_lang in tokenizer.lang_code_to_token else "hi"
        gen_kwargs["forced_bos_token_id"] = tokenizer.get_lang_id(tgt_code)

    all_hyps    = []
    all_refs    = []
    all_lengths = []

    pbar = tqdm(test_loader, desc=f"  Inference [{config_name}]",
                unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].clone()

            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            hyps = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            # decode references
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_hyps    += hyps
            all_refs    += refs
            all_lengths += batch["src_lengths"].tolist()
            pbar.set_postfix(decoded=len(all_hyps))

    # ── overall metrics ─────────────────────
    overall = {
        "bleu":  round(corpus_bleu(all_hyps, all_refs),  2),
        "chrf":  round(corpus_chrf(all_hyps, all_refs),  2),
        "ter":   round(corpus_ter(all_hyps,  all_refs),  2),
        "count": len(all_hyps),
    }
    log.info("Overall  BLEU=%.2f  chrF=%.2f  TER=%.2f  (#%d)",
             overall["bleu"], overall["chrf"], overall["ter"], overall["count"])

    # ── bin-wise metrics ────────────────────
    bins = bin_wise_metrics(all_hyps, all_refs, all_lengths)
    for bname in BIN_NAMES:
        b = bins[bname]
        log.info("  Bin %-6s  BLEU=%5.2f  chrF=%5.2f  TER=%5.2f  (#%d)",
                 bname, b["bleu"], b["chrf"], b["ter"], b["count"])

    # ── save predictions ────────────────────
    pred_file = os.path.join(PRED_DIR, f"{config_name}_predictions.tsv")
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("src_len\treference\thypothesis\n")
        for length, ref, hyp in zip(all_lengths, all_refs, all_hyps):
            f.write(f"{length}\t{ref}\t{hyp}\n")
    log.info("Predictions saved → %s", pred_file)

    # ── save metrics ────────────────────────
    metrics = {"overall": overall, "bins": bins}
    save_results(
        metrics,
        os.path.join(OUTPUT_DIR, "results", f"{config_name}_test_metrics.json"),
    )
    return metrics


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate fine-tuned HF models on the test set"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", choices=list(PRETRAINED_CONFIGS.keys()))
    g.add_argument("--group",  choices=list(GROUP_MAP.keys()))
    g.add_argument("--all",    action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    to_run = (list(PRETRAINED_CONFIGS.keys()) if args.all
              else GROUP_MAP[args.group] if args.group
              else [args.config])

    all_res = {}
    for name in to_run:
        res = evaluate_pretrained(name, args.batch_size)
        if res:
            all_res[name] = res

    if all_res:
        print("\n" + "="*80)
        print(f"  {'Model':<22} {'BLEU':>6} {'chrF':>6} {'TER':>6}  | Bin-wise BLEU →")
        hdr = "  ".join(f"{b:>6}" for b in BIN_NAMES)
        print(f"  {'':22} {'':>6} {'':>6} {'':>6}  | {hdr}")
        print("="*80)
        for name, res in all_res.items():
            ov = res["overall"]
            bv = "  ".join(f"{res['bins'][b]['bleu']:>6.2f}" for b in BIN_NAMES)
            print(f"  {name:<22} {ov['bleu']:>6.2f} {ov['chrf']:>6.2f} {ov['ter']:>6.2f}  | {bv}")
        print("="*80 + "\n")