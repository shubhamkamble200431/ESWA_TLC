import os, sys, json, logging, argparse
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    import sacrebleu; SACREBLEU_OK = True
except ImportError:
    SACREBLEU_OK = False; print("[WARN] pip install sacrebleu")

def _probe_comet_safe() -> bool:
    import subprocess, sys
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "from comet import download_model, load_from_checkpoint; print('ok')"],
            timeout=30, capture_output=True,
        )
        return r.returncode == 0
    except Exception:
        return False

if _probe_comet_safe():
    try:
        from comet import download_model, load_from_checkpoint as _comet_load
        COMET_OK = True
    except Exception:
        COMET_OK = False
else:
    COMET_OK = False

try:
    from peft import PeftModel; PEFT_OK = True
except ImportError:
    PEFT_OK = False; print("[WARN] pip install peft")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

COMBO_CASE_IDS = [
    "X-0000","X-1000","X-0100","X-0010","X-0001",
    "X-1100","X-1010","X-1001","X-0110","X-0101","X-0011",
    "X-1110","X-1101","X-1011","X-0111","X-1111",
]

ALL_CASE_IDS = COMBO_CASE_IDS

CASE_META = {
    "X-0000": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-1000": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-0100": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-0010": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-0001": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-1100": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-1010": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-1001": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-0110": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-0101": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-0011": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-1110": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-1101": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-1011": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-0111": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
    "X-1111": ("facebook/nllb-200-distilled-600M", "san_Deva", "hin_Deva", False),
}

CASE_DESC = {
    "X-0000": "Baseline (no extras)",
    "X-1000": "BBS",
    "X-0100": "LTC",
    "X-0010": "RoPE",
    "X-0001": "ALiBi",
    "X-1100": "BBS + LTC",
    "X-1010": "BBS + RoPE",
    "X-1001": "BBS + ALiBi",
    "X-0110": "LTC + RoPE",
    "X-0101": "LTC + ALiBi",
    "X-0011": "RoPE + ALiBi",
    "X-1110": "BBS + LTC + RoPE",
    "X-1101": "BBS + LTC + ALiBi",
    "X-1011": "BBS + RoPE + ALiBi",
    "X-0111": "LTC + RoPE + ALiBi",
    "X-1111": "BBS + LTC + RoPE + ALiBi",
}

BUCKET_BOUNDARIES = [5, 10, 15, 20]
BUCKET_LABELS = {1:"B1_1-5", 2:"B2_6-10", 3:"B3_11-15", 4:"B4_16-20", 5:"B5_20+"}

def assign_bucket(n):
    for i, b in enumerate(BUCKET_BOUNDARIES):
        if n <= b: return i + 1
    return 5


def _resolve_preset(raw: str) -> List[str]:
    r = raw.strip().lower()
    if r in ("all", ""):    return ALL_CASE_IDS[:]
    if r == "combo":        return COMBO_CASE_IDS[:]
    return []

def prompt_cases() -> List[str]:
    print("\n" + "=" * 70)
    print("  SAN→HIN NMT — Standalone Evaluator")
    print("=" * 70)
    print("  Combinatorial grid (X-BLRA  B=BBS L=LTC R=RoPE A=ALiBi):")
    for cid in COMBO_CASE_IDS:
        fl = cid[2:]
        flags = f"BBS={'✓' if fl[0]=='1' else '✗'} LTC={'✓' if fl[1]=='1' else '✗'} "
        flags += f"RoPE={'✓' if fl[2]=='1' else '✗'} ALiBi={'✓' if fl[3]=='1' else '✗'}"
        print(f"    {cid}   {CASE_DESC[cid]:<28}  {flags}")
    print()
    print("  Presets:  all | combo")
    print("  Or list case IDs space-separated: e.g.  X-0001 X-1111")
    print("  Press Enter for default: all\n")
    raw = input("  Cases [all]: ").strip()
    preset = _resolve_preset(raw)
    if preset:
        label = raw if raw else "all"
        print(f"  → Preset '{label}': {len(preset)} cases")
        return preset
    tokens = raw.upper().split()
    chosen = []
    for t in tokens:
        if t.startswith("X-"):   chosen.append(t)
    invalid = [c for c in chosen if c not in CASE_META]
    if invalid:
        print(f"  [WARN] Ignoring unknown IDs: {invalid}")
        chosen = [c for c in chosen if c in CASE_META]
    if not chosen:
        print("  No valid cases. Exiting."); sys.exit(0)
    print(f"  → Selected: {' '.join(chosen)}")
    return chosen


def load_data(data_dir, split, max_samples=None):
    sp = os.path.join(data_dir, f"{split}.sa")
    tp = os.path.join(data_dir, f"{split}.hi")
    for p in (sp, tp):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")
    src = [l.rstrip("\n") for l in open(sp, encoding="utf-8") if l.strip()]
    tgt = [l.rstrip("\n") for l in open(tp, encoding="utf-8") if l.strip()]
    if len(src) != len(tgt):
        raise ValueError(f"Line count mismatch: src={len(src)} tgt={len(tgt)}")
    if max_samples:
        src, tgt = src[:max_samples], tgt[:max_samples]
    logger.info(f"Loaded {len(src):,} {split} pairs from {data_dir}")
    return src, tgt


def _is_lora(path):
    return os.path.exists(os.path.join(path, "adapter_config.json"))

def load_model_and_tokenizer(checkpoint_path, case_id, device="cuda"):
    base_id, src_lang, tgt_lang, nominal_lora = CASE_META[case_id]
    is_lora_ckpt = nominal_lora and _is_lora(checkpoint_path) and PEFT_OK

    tok_kw = {"use_fast": True}
    if "nllb" in base_id or "m2m" in base_id:
        tok_kw.update(src_lang=src_lang, tgt_lang=tgt_lang)
    logger.info(f"Loading tokenizer: {checkpoint_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, **tok_kw)
    except Exception as e:
        logger.warning(f"Checkpoint tokenizer failed ({e}), using base model.")
        tokenizer = AutoTokenizer.from_pretrained(base_id, **tok_kw)
    tok_size = len(tokenizer)
    logger.info(f"Tokenizer vocab size: {tok_size:,}")

    if is_lora_ckpt:
        logger.info(f"Loading base model from HF hub: {base_id}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_id, dtype=torch.float32, low_cpu_mem_usage=True)

        emb_size = model.get_input_embeddings().weight.shape[0]
        if emb_size != tok_size:
            logger.info(f"Resizing embeddings: {emb_size:,} → {tok_size:,} "
                        f"(+{tok_size - emb_size} tokens added during training)")
            model.resize_token_embeddings(tok_size)

        logger.info(f"Loading LoRA adapter: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
        logger.info("Merging LoRA into base model…")
        model = model.merge_and_unload()
        logger.info("Merge complete.")

    else:
        logger.info(f"Loading plain checkpoint: {checkpoint_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path, dtype=torch.float16, low_cpu_mem_usage=True)
        emb_size = model.get_input_embeddings().weight.shape[0]
        if emb_size != tok_size:
            logger.info(f"Resizing embeddings: {emb_size:,} → {tok_size:,}")
            model.resize_token_embeddings(tok_size)

    model = model.to(device).eval()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model on {device} | {total:.1f}M params")
    return model, tokenizer


def generate(model, tokenizer, sources, case_id,
             batch_size=16, max_length=256, num_beams=4, device="cuda"):
    base_id, src_lang, tgt_lang, _ = CASE_META[case_id]
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang

    forced_bos = None
    if any(x in base_id for x in ("nllb", "m2m", "mbart")):
        if hasattr(tokenizer, "lang_code_to_id"):
            forced_bos = tokenizer.lang_code_to_id.get(tgt_lang)
            if forced_bos:
                logger.info(f"forced_bos_token_id={forced_bos} ('{tgt_lang}')")
            else:
                logger.warning(f"'{tgt_lang}' not in lang_code_to_id! "
                               f"Sample keys: {list(tokenizer.lang_code_to_id)[:6]}")

    preds, n = [], (len(sources) + batch_size - 1) // batch_size
    for i in range(0, len(sources), batch_size):
        batch  = sources[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        gen_kw = dict(max_length=max_length, num_beams=num_beams,
                      early_stopping=True, no_repeat_ngram_size=3)
        if forced_bos is not None:
            gen_kw["forced_bos_token_id"] = forced_bos
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        bn = i // batch_size + 1
        if bn % 20 == 0 or bn == n:
            logger.info(f"  [{bn:>4}/{n}] {min(i+batch_size, len(sources)):,}/{len(sources):,}")
    return preds


_COMET_CACHE = None

def _comet_score(hyps, refs, srcs):
    global _COMET_CACHE
    if not COMET_OK: return None
    try:
        if _COMET_CACHE is None:
            logger.info("[COMET] Loading model…")
            _COMET_CACHE = _comet_load(download_model("Unbabel/wmt22-comet-da"))
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        sc   = _COMET_CACHE.predict(data, batch_size=64,
                                    gpus=1 if torch.cuda.is_available() else 0)
        return float(np.mean(sc.scores))
    except Exception as e:
        logger.warning(f"[COMET] {e}"); return None

def _bleu(h, r):
    if not SACREBLEU_OK or not h: return 0.0
    return sacrebleu.corpus_bleu(h, [r], tokenize="flores200").score

def _chrf(h, r):
    if not SACREBLEU_OK or not h: return 0.0
    return sacrebleu.corpus_chrf(h, [r], word_order=2).score

def _lr(h, r):
    if not h: return 0.0
    return float(np.mean([len(a.split())/max(len(b.split()),1) for a,b in zip(h,r)]))

def compute_metrics(preds, refs, sources, run_comet=True):
    buckets = [assign_bucket(len(s.split())) for s in sources]
    res = {
        "bleu":         _bleu(preds, refs),
        "chrf":         _chrf(preds, refs),
        "length_ratio": _lr(preds, refs),
    }
    bucket_bleus = {}
    for b in range(1, 6):
        idx = [i for i, bk in enumerate(buckets) if bk == b]
        lbl = BUCKET_LABELS[b]
        if not idx:
            logger.warning(f"  No samples in {lbl}"); continue
        bh = [preds[i] for i in idx]; br = [refs[i] for i in idx]
        res[f"bleu_{lbl}"]         = _bleu(bh, br)
        res[f"chrf_{lbl}"]         = _chrf(bh, br)
        res[f"length_ratio_{lbl}"] = _lr(bh, br)
        bucket_bleus[b] = res[f"bleu_{lbl}"]
        logger.info(f"  {lbl:<12} n={len(idx):>5}  BLEU={res[f'bleu_{lbl}']:6.2f}  "
                    f"chrF={res[f'chrf_{lbl}']:6.2f}  LR={res[f'length_ratio_{lbl}']:.3f}")
    if run_comet:
        s = _comet_score(preds, refs, sources)
        if s is not None: res["comet"] = s
    return res


def print_report(metrics, case_id, checkpoint, split):
    W = 72
    print("\n" + "="*W)
    print(f"  {case_id}: {CASE_DESC.get(case_id,'')}  |  {split}")
    print(f"  {checkpoint}")
    print("="*W)
    for k in ["bleu","chrf","comet","length_ratio"]:
        if k in metrics:
            print(f"  {k:<28} {metrics[k]:>10.4f}")
    print()
    for b in range(1, 6):
        lbl = BUCKET_LABELS[b]
        if f"bleu_{lbl}" in metrics:
            print(f"  {lbl:<14}"
                  f"  BLEU={metrics[f'bleu_{lbl}']:6.2f}"
                  f"  chrF={metrics.get(f'chrf_{lbl}',0):6.2f}"
                  f"  LR={metrics.get(f'length_ratio_{lbl}',0):.3f}")
    print("="*W + "\n")

def save_outputs(metrics, preds, sources, refs, case_id, split, checkpoint, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jp = os.path.join(out_dir, f"{case_id}_{split}_metrics_{ts}.json")
    json.dump({"case_id":case_id,"split":split,"checkpoint":checkpoint,
               "timestamp":ts,"metrics":metrics},
              open(jp,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
    logger.info(f"Metrics  → {jp}")

    pp = os.path.join(out_dir, f"{case_id}_{split}_predictions_{ts}.hi")
    open(pp,"w",encoding="utf-8").write("\n".join(preds)+"\n")
    logger.info(f"Preds    → {pp}")

    buckets = [assign_bucket(len(s.split())) for s in sources]
    ep = os.path.join(out_dir, f"{case_id}_{split}_examples_{ts}.txt")
    seen = {b:0 for b in range(1,6)}
    with open(ep,"w",encoding="utf-8") as f:
        f.write(f"SAN→HIN | {case_id}: {CASE_DESC.get(case_id,'')} | {split} | {ts}\n")
        f.write("="*80+"\n\n")
        for i,(src,ref,hyp,bk) in enumerate(zip(sources,refs,preds,buckets)):
            if seen[bk] >= 10: continue
            f.write(f"[{BUCKET_LABELS[bk]}]  idx={i}\n"
                    f"  SRC: {src}\n  REF: {ref}\n  HYP: {hyp}\n\n")
            seen[bk] += 1
    logger.info(f"Examples → {ep}")


def run_case(case_id, checkpoint, data_dir, split="test",
             batch_size=16, num_beams=4, max_length=256,
             out_dir=None, device="cuda", max_samples=None, run_comet=True):
    logger.info(f"\n{'='*60}\n  {case_id}: {CASE_DESC.get(case_id,'')}\n"
                f"  ckpt = {checkpoint}\n{'='*60}")
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    sources, refs = load_data(data_dir, split, max_samples)
    model, tok    = load_model_and_tokenizer(checkpoint, case_id, device)

    logger.info("Generating translations…")
    preds = generate(model, tok, sources, case_id,
                     batch_size=batch_size, max_length=max_length,
                     num_beams=num_beams, device=device)

    logger.info("\nSample (first 3):")
    for i in range(min(3, len(preds))):
        logger.info(f"  SRC: {sources[i]}")
        logger.info(f"  REF: {refs[i]}")
        logger.info(f"  HYP: {preds[i]}\n")

    logger.info("Computing metrics…")
    metrics = compute_metrics(preds, refs, sources, run_comet=run_comet)
    print_report(metrics, case_id, checkpoint, split)

    od = out_dir or os.path.join(os.path.dirname(checkpoint), "eval_results")
    save_outputs(metrics, preds, sources, refs, case_id, split, checkpoint, od)

    del model; torch.cuda.empty_cache()
    return metrics


def print_summary(all_results):
    W = 100
    print("\n" + "="*W)
    print("  FINAL SUMMARY — SAN→HIN NMT")
    print("="*W)
    hdr = (f"  {'Case':<10}  {'BLEU':>7}  {'chrF':>7}  {'COMET':>7}  "
           + "  ".join(f"{'B'+str(b):>6}" for b in range(1,6))
           + "  Description")
    print(hdr); print("-"*W)
    evaluated = [c for c in COMBO_CASE_IDS if c in all_results]
    if evaluated:
        print(f"  ── Combo ──────")
        for cid in evaluated:
            m = all_results[cid]
            if "error" in m:
                print(f"  {cid:<10}  ERROR: {str(m['error'])[:60]}"); continue
            bv   = "  ".join(f"{m.get(f'bleu_{BUCKET_LABELS[b]}',0):6.2f}" for b in range(1,6))
            desc = CASE_DESC.get(cid, "")[:30]
            print(f"  {cid:<10}  {m.get('bleu',0):7.2f}  {m.get('chrf',0):7.2f}  "
                  f"{m.get('comet',0):7.3f}  {bv}  {desc}")
    print("="*W+"\n")

def save_summary(all_results, out_dir, split):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p  = os.path.join(out_dir, f"summary_all_{split}_{ts}.json")
    json.dump(all_results, open(p,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
    logger.info(f"Summary → {p}")


def main():
    pa = argparse.ArgumentParser(
        description="Standalone evaluator — SAN→HIN NMT X-0000 through X-1111",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--cases", nargs="*", default=None,
        help="Case IDs (e.g. X-1111) or preset: all/combo. Omit for interactive.")
    pa.add_argument("--checkpoint_root",
        default="outputs",
        help="Root dir containing X-0000/, X-1111/, … subdirs with checkpoint-best/")
    pa.add_argument("--checkpoint", default=None,
        help="Explicit checkpoint path (single-case override).")
    pa.add_argument("--data_dir",   default="data")
    pa.add_argument("--split",      default="test_7264_5")
    pa.add_argument("--batch_size", type=int, default=16)
    pa.add_argument("--num_beams",  type=int, default=4)
    pa.add_argument("--max_length", type=int, default=256)
    pa.add_argument("--out_dir",    default=None,
        help="Override output dir (default: <checkpoint>/../eval_results)")
    pa.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--max_samples", type=int, default=None)
    pa.add_argument("--no_comet",  action="store_true",
        help="Skip COMET metric (~3 min/case).")
    args = pa.parse_args()

    if args.cases is None:
        chosen = prompt_cases()
    elif len(args.cases) == 1:
        preset = _resolve_preset(args.cases[0])
        if preset:
            chosen = preset
        else:
            raw = args.cases[0].upper()
            chosen = [raw if raw.startswith("X-") else raw]
            bad = [c for c in chosen if c not in CASE_META]
            if bad: pa.error(f"Unknown: {bad}. Valid: {ALL_CASE_IDS}")
    else:
        chosen = []
        for x in args.cases:
            u = x.upper()
            if u.startswith("X-"): chosen.append(u)
            else:                   chosen.append(u)
        bad = [c for c in chosen if c not in CASE_META]
        if bad: pa.error(f"Unknown cases: {bad}. Valid: {ALL_CASE_IDS}")

    run_comet = (not args.no_comet) and COMET_OK
    if not SACREBLEU_OK:
        logger.warning("sacrebleu not installed — BLEU/chrF will be 0.")

    print(f"\n  Cases    : {' '.join(chosen)}")
    print(f"  Split    : {args.split}")
    print(f"  Beams    : {args.num_beams}  |  Batch: {args.batch_size}")
    print(f"  COMET    : {run_comet}")
    print(f"  Device   : {args.device}\n")

    all_results = {}
    for cid in chosen:
        if args.checkpoint and len(chosen) == 1:
            ckpt = args.checkpoint
        else:
            ckpt = os.path.join(args.checkpoint_root, cid, "checkpoint-best")

        od = args.out_dir or os.path.join(args.checkpoint_root, cid, "eval_results")
        try:
            m = run_case(case_id=cid, checkpoint=ckpt, data_dir=args.data_dir,
                         split=args.split, batch_size=args.batch_size,
                         num_beams=args.num_beams, max_length=args.max_length,
                         out_dir=od, device=args.device,
                         max_samples=args.max_samples, run_comet=run_comet)
            all_results[cid] = m
        except Exception as e:
            logger.error(f"[{cid}] FAILED: {e}", exc_info=True)
            all_results[cid] = {"error": str(e)}

    if len(chosen) > 1:
        print_summary(all_results)
        sd = args.out_dir or os.path.join(args.checkpoint_root, "eval_summary")
        save_summary(all_results, sd, args.split)

if __name__ == "__main__":
    main()
