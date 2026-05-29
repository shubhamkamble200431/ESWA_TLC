import os
import sys
import json
import argparse
import logging

import torch
import numpy as np
from transformers import MT5ForConditionalGeneration, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import sacrebleu; SACREBLEU_OK = True
except ImportError:
    SACREBLEU_OK = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TASK_PREFIX = "translate Sanskrit to Hindi: "
BUCKET_BOUNDARIES = [5, 10, 15, 20]
BUCKET_LABELS = {1: "B1_1-5", 2: "B2_6-10", 3: "B3_11-15", 4: "B4_16-20", 5: "B5_20+"}


def assign_bucket(n):
    for i, b in enumerate(BUCKET_BOUNDARIES):
        if n <= b: return i + 1
    return 5


def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def generate(model, tokenizer, sources, batch_size, max_length, num_beams, device):
    preds = []
    for i in range(0, len(sources), batch_size):
        batch  = [TASK_PREFIX + s for s in sources[i: i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=max_length, num_beams=num_beams,
                                 early_stopping=True)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        if (i // batch_size) % 20 == 0:
            logger.info(f"  [{min(i+batch_size, len(sources))}/{len(sources)}]")
    return preds


def compute_metrics(preds, refs, sources):
    buckets = [assign_bucket(len(s.split())) for s in sources]
    res = {}
    if SACREBLEU_OK:
        res["bleu"] = sacrebleu.corpus_bleu(preds, [refs], tokenize="flores200").score
        res["chrf"] = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
        bucket_bleus = {}
        for b in range(1, 6):
            idx = [i for i, bk in enumerate(buckets) if bk == b]
            if not idx: continue
            lbl = BUCKET_LABELS[b]
            bp  = [preds[i] for i in idx]
            br  = [refs[i]  for i in idx]
            res[f"bleu_{lbl}"] = sacrebleu.corpus_bleu(bp, [br], tokenize="flores200").score
            res[f"chrf_{lbl}"] = sacrebleu.corpus_chrf(bp, [br], word_order=2).score
            bucket_bleus[b] = res[f"bleu_{lbl}"]
    return res


def main():
    parser = argparse.ArgumentParser(description="mT5 SAN→HIN inference and evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams",  type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--out_dir",    default=None)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    src_file = "test_7264_5.sa" if args.split == "test" else "val.sa"
    tgt_file = "test_7264_5.hi" if args.split == "test" else "val.hi"
    sources = load_lines(os.path.join(args.data_dir, src_file))
    refs    = load_lines(os.path.join(args.data_dir, tgt_file))
    logger.info(f"Loaded {len(sources):,} {args.split} pairs")

    logger.info(f"Loading model: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = MT5ForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=torch.float16, low_cpu_mem_usage=True).to(args.device).eval()

    logger.info("Generating translations...")
    preds = generate(model, tokenizer, sources, args.batch_size,
                     args.max_length, args.num_beams, args.device)

    logger.info("Computing metrics...")
    metrics = compute_metrics(preds, refs, sources)

    print(f"\n{'='*60}")
    print(f"  mT5  |  split={args.split}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"{'='*60}")
    for k in ["bleu", "chrf"] + [f"bleu_{BUCKET_LABELS[b]}" for b in range(1, 6)]:
        if k in metrics:
            print(f"  {k:<35} {metrics[k]:>8.2f}")
    print(f"{'='*60}\n")

    out_dir = args.out_dir or os.path.join(args.checkpoint, "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"mT5_{args.split}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"mT5_{args.split}_predictions.hi"), "w") as f:
        f.write("\n".join(preds) + "\n")
    logger.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
