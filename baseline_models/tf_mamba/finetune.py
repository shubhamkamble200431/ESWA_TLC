"""
finetune.py — Fine-tuning loop for pretrained HuggingFace models:
              M2M100  (facebook/m2m100_418M  |  facebook/m2m100_1.2B)
              LongT5  (google/long-t5-tglobal-base  |  -large)
              mT5     (google/mt5-small  |  -base  |  -large)

Usage:
    python finetune.py --config m2m100_418m
    python finetune.py --config mt5_base
    python finetune.py --all              # all pretrained models sequentially
    python finetune.py --group m2m100     # just M2M100 variants
    python finetune.py --group longt5
    python finetune.py --group mt5
"""

import argparse
import logging
import math
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import (
    ALL_PRETRAINED_CONFIGS, PRETRAINED_CONFIGS,
    CKPT_DIR, OUTPUT_DIR, DEVICE, SEED, PATIENCE,
)
from pretrained_dataset import load_hf_tokenizer, get_pretrained_dataloaders
from utils import set_seed, setup_logging, corpus_bleu, save_results

logger = logging.getLogger(__name__)

# ── Group aliases ────────────────────────────────────────────────
GROUP_MAP = {
    "m2m100": [k for k, v in PRETRAINED_CONFIGS.items() if v.model_type == "m2m100"],
    "longt5": [k for k, v in PRETRAINED_CONFIGS.items() if v.model_type == "longt5"],
    "mt5":    [k for k, v in PRETRAINED_CONFIGS.items() if v.model_type == "mt5"],
    "all":    list(PRETRAINED_CONFIGS.keys()),
}


# ─────────────────────────────────────────────────────────────────
# HuggingFace model loader
# ─────────────────────────────────────────────────────────────────

def load_hf_model(cfg):
    """Load the correct HF model class for the given config."""
    from transformers import (
        M2M100ForConditionalGeneration,
        AutoModelForSeq2SeqLM,
    )
    if cfg.model_type == "m2m100":
        model = M2M100ForConditionalGeneration.from_pretrained(cfg.hf_model_id)
    else:
        # mt5 and longt5 both work with AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.hf_model_id)

    logger.info(
        "Loaded %s  (%s)  — params: %s",
        cfg.hf_model_id, cfg.model_type,
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    return model


# ─────────────────────────────────────────────────────────────────
# Decoding helper
# ─────────────────────────────────────────────────────────────────

def batch_generate(model, batch, tokenizer, cfg, device) -> list:
    """Generate translations for a batch; returns list of decoded strings."""
    gen_kwargs = dict(
        max_new_tokens=cfg.max_tgt_len,
        num_beams=cfg.num_beams,
    )
    if cfg.model_type == "m2m100":
        # get_lang_id raises KeyError for unsupported codes; use hi as fallback
        tgt_code = cfg.tgt_lang if cfg.tgt_lang in tokenizer.lang_code_to_token else "hi"
        gen_kwargs["forced_bos_token_id"] = tokenizer.get_lang_id(tgt_code)

    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    steps      = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:>3} [train]", unit="batch",
                leave=False, dynamic_ncols=True)

    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()

        if cfg.fp16 and scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps      += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / max(steps, 1)


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate(model, loader, tokenizer, device, cfg, epoch: int):
    model.eval()
    total_loss = 0.0
    steps      = 0
    hyps, refs = [], []

    pbar = tqdm(loader, desc=f"  Epoch {epoch:>3} [val]  ", unit="batch",
                leave=False, dynamic_ncols=True)

    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        total_loss += outputs.loss.item()
        steps      += 1
        pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        # decode for BLEU
        preds = batch_generate(model, batch, tokenizer, cfg, device)
        hyps += preds

        # decode references (replace -100 with pad_id first)
        lbl_ids = labels.clone()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id
        refs += tokenizer.batch_decode(lbl_ids, skip_special_tokens=True)

    val_loss = total_loss / max(steps, 1)
    val_bleu = corpus_bleu(hyps, refs)
    return val_loss, val_bleu


# ─────────────────────────────────────────────────────────────────
# Warmup + linear-decay scheduler
# ─────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, _lr_lambda)


# ─────────────────────────────────────────────────────────────────
# Main fine-tuning function
# ─────────────────────────────────────────────────────────────────

def finetune_model(config_name: str) -> float:
    set_seed(SEED)
    log = setup_logging(f"finetune_{config_name}")
    log.info("=" * 60)
    log.info("Fine-tuning: %s", config_name)

    cfg    = ALL_PRETRAINED_CONFIGS[config_name]
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.fp16 and torch.cuda.is_available()
    log.info("Device: %s  |  fp16: %s", device, use_fp16)

    # ── tokeniser + data ────────────────────
    tokenizer = load_hf_tokenizer(cfg)
    train_loader, val_loader, _ = get_pretrained_dataloaders(cfg, tokenizer)

    # ── model ───────────────────────────────
    model = load_hf_model(cfg).to(device)

    # ── optimiser + scheduler ───────────────
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    total_steps  = len(train_loader) * cfg.max_epochs
    warmup_steps = math.ceil(total_steps * cfg.warmup_ratio)
    scheduler    = get_scheduler(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler() if use_fp16 else None

    log.info("Total steps: %d  |  Warmup: %d", total_steps, warmup_steps)

    ckpt_path    = os.path.join(CKPT_DIR, f"{config_name}_best")
    best_bleu    = 0.0
    patience_ctr = 0
    history      = {"train_loss": [], "val_loss": [], "val_bleu": []}

    epoch_bar = tqdm(range(1, cfg.max_epochs + 1), desc=f"[{config_name}]",
                     unit="epoch", dynamic_ncols=True)

    for epoch in epoch_bar:
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 scaler, device, cfg, epoch)
        val_loss, val_bleu = validate(model, val_loader, tokenizer, device, cfg, epoch)
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bleu"].append(val_bleu)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_bleu=f"{val_bleu:.2f}",
            best=f"{best_bleu:.2f}",
        )
        log.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f "
            "val_bleu=%.2f lr=%.2e time=%.1fs",
            epoch, cfg.max_epochs,
            train_loss, val_loss, val_bleu,
            optimizer.param_groups[0]["lr"], elapsed,
        )

        if val_bleu > best_bleu:
            best_bleu    = val_bleu
            patience_ctr = 0
            # save HF-style checkpoint (full model + tokenizer)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            log.info("  ✓ Best checkpoint saved → %s  (BLEU %.2f)", ckpt_path, best_bleu)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info("Early stopping at epoch %d.", epoch)
                break

    save_results(
        history,
        os.path.join(OUTPUT_DIR, "results", f"{config_name}_train_history.json"),
    )
    log.info("Done. Best val BLEU: %.2f", best_bleu)
    return best_bleu


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune pretrained HF models for Sanskrit→Hindi NMT"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", choices=list(PRETRAINED_CONFIGS.keys()),
                   help="Single pretrained config to fine-tune")
    g.add_argument("--group",  choices=list(GROUP_MAP.keys()),
                   help="Fine-tune a named group: m2m100 | longt5 | mt5 | all")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    to_run  = [args.config] if args.config else GROUP_MAP[args.group]
    results = {}

    print(f"\n{'='*60}")
    print(f"  Fine-tuning {len(to_run)} model(s): {to_run}")
    print(f"{'='*60}\n")

    for name in to_run:
        bleu           = finetune_model(name)
        results[name]  = {"val_bleu": round(bleu, 2)}

    print("\n" + "="*60)
    print("  Fine-tuning Summary — Best Val BLEU")
    print("="*60)
    for name, r in results.items():
        print(f"  {name:<22}  BLEU = {r['val_bleu']:.2f}")
    print("="*60 + "\n")