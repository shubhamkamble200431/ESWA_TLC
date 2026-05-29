import os
import json
import logging
import argparse
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from configs.combo_configs import get_combo_config as get_config
from data.dataset import load_parallel_corpus
from data.transliterate import transliterate_corpus
from evaluation.metrics import compute_all_metrics, BUCKET_LABELS, assign_bucket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    checkpoint_path: str,
    case_id: str,
    split: str = "test",
    batch_size: int = 32,
    num_beams: int = 4,
    device: str = "cuda",
    max_samples: int = None,
) -> dict:
    cfg = get_config(case_id)

    src_file = f"{split}.sa"
    tgt_file = f"{split}.hi"
    src_sents, tgt_sents = load_parallel_corpus(
        cfg.data_dir, src_file, tgt_file, max_samples=max_samples
    )

    if cfg.transliterate_to_devanagari:
        src_sents = transliterate_corpus(src_sents)
        logger.info("Applied Sanskrit variant normalisation.")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id.get(cfg.tgt_lang)

    predictions = _batch_generate(
        model, tokenizer, src_sents,
        src_lang=cfg.src_lang,
        batch_size=batch_size,
        max_length=min(cfg.max_target_length, 128),
        num_beams=num_beams,
        forced_bos_token_id=forced_bos_token_id,
        device=device,
    )

    metrics = compute_all_metrics(predictions, tgt_sents, sources=src_sents)

    _print_report(metrics, case_id, checkpoint_path, split)

    out_dir = os.path.join(cfg.output_dir, "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"{case_id}_{split}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"case_id": case_id, "split": split, "metrics": metrics}, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    ex_path = os.path.join(out_dir, f"{case_id}_{split}_{ts}_examples.txt")
    with open(ex_path, "w", encoding="utf-8") as f:
        for i in range(min(20, len(src_sents))):
            b = assign_bucket(len(src_sents[i].split()))
            f.write(f"[B{b}] SRC: {src_sents[i]}\n")
            f.write(f"      REF: {tgt_sents[i]}\n")
            f.write(f"      HYP: {predictions[i]}\n\n")

    return metrics


def _batch_generate(
    model, tokenizer, sources: list,
    src_lang: str, batch_size: int,
    max_length: int, forced_bos_token_id, device: str, num_beams: int = 4,
) -> list:
    all_preds = []
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang

    for i in range(0, len(sources), batch_size):
        batch = sources[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "early_stopping": True,
        }
        if forced_bos_token_id:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_preds.extend(decoded)

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Generated {min(i+batch_size, len(sources))}/{len(sources)}")

    return all_preds


def _print_report(metrics: dict, case_id: str, checkpoint: str, split: str):
    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT — {case_id}  |  split={split}")
    print(f"  Checkpoint: {checkpoint}")
    print("=" * 70)
    print(f"  {'Metric':<35} {'Score':>10}")
    print("-" * 70)

    priority = ["bleu", "chrf", "comet", "length_ratio"]
    for key in priority:
        if key in metrics:
            print(f"  {key:<35} {metrics[key]:>10.2f}")

    print()
    for b in range(1, 6):
        label = BUCKET_LABELS.get(b, f"B{b}")
        bleu_key = f"bleu_{label}"
        if bleu_key in metrics:
            print(
                f"  {bleu_key:<35} {metrics[bleu_key]:>10.2f}"
                f"  chrf={metrics.get(f'chrf_{label}', 0):.1f}"
                f"  lr={metrics.get(f'length_ratio_{label}', 0):.3f}"
            )

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NMT checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint")
    parser.add_argument("--case_id",    required=True, help="X-0000 through X-1111")
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        case_id=args.case_id,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
    )
