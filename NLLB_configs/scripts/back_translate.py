import os
import argparse
import logging
import math

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def back_translate(
    hi_mono_file: str,
    out_dir: str,
    model_name: str = "facebook/nllb-200-distilled-600M",
    batch_size: int = 16,
    confidence_threshold: float = -2.5,
    max_lines: int = 50_000,
    device: str = "cuda",
):
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="hin_Deva")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    forced_bos_id = tokenizer.lang_code_to_id.get("san_Deva")
    if not forced_bos_id:
        forced_bos_id = tokenizer.lang_code_to_id.get("san_Deva")
        logger.warning(
            "san_Deva not in vocab, using san_Deva for BT. "
            "Results will be in transliterated Sanskrit."
        )

    with open(hi_mono_file, "r", encoding="utf-8") as f:
        hi_lines = [l.rstrip() for l in f][:max_lines]

    logger.info(f"Back-translating {len(hi_lines):,} Hindi sentences → SAN")

    sa_lines, kept_hi = [], []
    accepted = rejected = 0

    for i in range(0, len(hi_lines), batch_size):
        batch_hi = hi_lines[i: i + batch_size]
        inputs = tokenizer(
            batch_hi, return_tensors="pt",
            padding=True, truncation=True, max_length=256,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=300,
                num_beams=4,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        sequences = outputs.sequences
        scores    = outputs.sequences_scores

        batch_sa = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        for hi_sent, sa_sent, score in zip(batch_hi, batch_sa, scores.tolist()):
            avg_lp = score / max(len(sa_sent.split()), 1)
            if avg_lp >= confidence_threshold:
                sa_lines.append(sa_sent)
                kept_hi.append(hi_sent)
                accepted += 1
            else:
                rejected += 1

        if i % 1000 == 0:
            logger.info(
                f"  {i+batch_size}/{len(hi_lines)} | "
                f"accepted={accepted} | rejected={rejected}"
            )

    logger.info(f"BT complete: kept {accepted:,} / {len(hi_lines):,} pairs.")

    sa_path = os.path.join(out_dir, "bt.sa")
    hi_path = os.path.join(out_dir, "bt.hi")
    with open(sa_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sa_lines))
    with open(hi_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept_hi))

    logger.info(f"Saved: {sa_path}  ({len(sa_lines):,} lines)")
    logger.info(f"Saved: {hi_path}  ({len(kept_hi):,} lines)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hi_mono_file",        required=True)
    parser.add_argument("--out_dir",             default="65k_data/bt_augmented")
    parser.add_argument("--model",               default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--batch_size",          type=int,   default=16)
    parser.add_argument("--confidence_threshold",type=float, default=-2.5)
    parser.add_argument("--max_lines",           type=int,   default=50_000)
    parser.add_argument("--device",              default="cuda")
    args = parser.parse_args()

    back_translate(
        hi_mono_file=args.hi_mono_file,
        out_dir=args.out_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        max_lines=args.max_lines,
        device=args.device,
    )
