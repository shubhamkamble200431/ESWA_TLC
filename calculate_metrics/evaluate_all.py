import os, sys, glob, csv, json, argparse, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_PRED_DIR  = "/media/kpdubey/8.0 TB Volume/Shubham/LCS/calculate_metrics_65k/prediction"
DEFAULT_OUT_DIR   = "1452"
DEFAULT_SRC_DIR   = None
DEFAULT_SRC_FILE  = "/media/kpdubey/8.0 TB Volume/Shubham/LCS/calculate_metrics_65k/test_7264_5.sa"
DEFAULT_SPLIT     = "test"

BUCKET_BOUNDARIES = [5, 10, 15, 20]
BUCKET_LABELS     = {1: "B1_1-5", 2: "B2_6-10", 3: "B3_11-15",
                     4: "B4_16-20", 5: "B5_21+"}

def assign_bucket(n: int) -> int:
    for i, b in enumerate(BUCKET_BOUNDARIES):
        if n <= b:
            return i + 1
    return 5

try:
    import sacrebleu; SACREBLEU_OK = True
except ImportError:
    SACREBLEU_OK = False; logger.warning("[WARN] pip install sacrebleu")

try:
    from bert_score import score as bert_score_fn; BERTSCORE_OK = True
except ImportError:
    BERTSCORE_OK = False; logger.warning("[WARN] pip install bert-score")

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try: nltk.download(pkg, quiet=True)
        except: pass
    METEOR_OK = True
except ImportError:
    METEOR_OK = False; logger.warning("[WARN] pip install nltk")

try:
    import torch
    from comet import download_model, load_from_checkpoint as comet_load
    COMET_OK = True
except ImportError:
    COMET_OK = False; logger.warning("[WARN] pip install unbabel-comet")


def read_tsv(path: str) -> Tuple[List[int], List[str], List[str]]:
    src_lens, refs, hyps = [], [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader, 1):
            try:
                src_lens.append(int(row["src_len"]))
                refs.append(row["reference"].strip())
                hyps.append(row["hypothesis"].strip())
            except (KeyError, ValueError) as e:
                logger.warning(f"  Skipping row {i} in {path}: {e}")
    logger.info(f"  Loaded {len(refs):,} rows from {Path(path).name}")
    return src_lens, refs, hyps


def resolve_source_file(src_file: Optional[str],
                        src_dir: Optional[str],
                        split: str) -> Optional[str]:
    if src_file:
        if os.path.exists(src_file):
            return src_file
        logger.warning(f"src_file not found: {src_file}")
        return None

    if src_dir:
        p = os.path.join(src_dir, f"{split}.sa")
        if os.path.exists(p):
            return p
        logger.warning(f"Source file not found: {p} — COMET will be skipped.")
        return None

    return None


def read_source_lines(path: str) -> Optional[List[str]]:
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]
    logger.info(f"Loaded {len(lines):,} source sentences from {path}")
    return lines


def _safe_hyps(hyps, refs):
    h2, r2 = zip(*[(h, r) for h, r in zip(hyps, refs) if h.strip() and r.strip()]) \
        if any(h.strip() and r.strip() for h, r in zip(hyps, refs)) else ([], [])
    return list(h2), list(r2)

def metric_bleu(hyps, refs):
    if not SACREBLEU_OK or not hyps: return float("nan")
    h, r = _safe_hyps(hyps, refs)
    return sacrebleu.corpus_bleu(h, [r], tokenize="flores200").score

def metric_chrf(hyps, refs):
    if not SACREBLEU_OK or not hyps: return float("nan")
    h, r = _safe_hyps(hyps, refs)
    return sacrebleu.corpus_chrf(h, [r], word_order=2).score

def metric_chrf2(hyps, refs):
    if not SACREBLEU_OK or not hyps: return float("nan")
    h, r = _safe_hyps(hyps, refs)
    return sacrebleu.corpus_chrf(h, [r], beta=2).score

def metric_ter(hyps, refs):
    if not SACREBLEU_OK or not hyps: return float("nan")
    h, r = _safe_hyps(hyps, refs)
    if not h: return float("nan")
    try:
        return sacrebleu.corpus_ter(h, [r]).score
    except Exception as e:
        logger.warning(f"  TER failed: {e}")
        return float("nan")

def metric_bertscore(hyps, refs):
    if not BERTSCORE_OK or not hyps:
        return float("nan"), float("nan"), float("nan")
    h, r = _safe_hyps(hyps, refs)
    if not h:
        return float("nan"), float("nan"), float("nan")
    try:
        P, R, F1 = bert_score_fn(h, r, lang="hi", model_type="xlm-roberta-base",
                                  verbose=False)
        return float(P.mean()), float(R.mean()), float(F1.mean())
    except Exception as e:
        logger.warning(f"  BERTScore failed: {e}")
        return float("nan"), float("nan"), float("nan")

def metric_meteor(hyps, refs):
    if not METEOR_OK or not hyps: return float("nan")
    scores = []
    for h, r in zip(hyps, refs):
        if not h.strip() or not r.strip(): continue
        try:
            scores.append(meteor_score([word_tokenize(r)], word_tokenize(h)))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else float("nan")

_COMET_MODEL_CACHE = None

def metric_comet(hyps, refs, srcs):
    global _COMET_MODEL_CACHE
    if not COMET_OK or not hyps or not srcs: return float("nan")
    try:
        if _COMET_MODEL_CACHE is None:
            logger.info("  [COMET] Downloading/loading model...")
            _COMET_MODEL_CACHE = comet_load(
                download_model("Unbabel/wmt22-comet-da"))
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(srcs, hyps, refs)
                if h.strip() and r.strip() and s.strip()]
        import torch
        gpus = 1 if torch.cuda.is_available() else 0
        out  = _COMET_MODEL_CACHE.predict(data, batch_size=32, gpus=gpus)
        return float(np.mean(out.scores))
    except Exception as e:
        logger.warning(f"  [COMET] {e}")
        return float("nan")


def compute_all_metrics(hyps, refs, srcs=None,
                        run_bertscore=True, run_meteor=True, run_comet=True,
                        label="") -> Dict[str, float]:
    res: Dict[str, float] = {}

    logger.info(f"  {label}: BLEU...")
    res["BLEU"]  = metric_bleu(hyps, refs)

    logger.info(f"  {label}: chrF++...")
    res["chrF"]  = metric_chrf(hyps, refs)

    logger.info(f"  {label}: chrF2...")
    res["chrF2"] = metric_chrf2(hyps, refs)

    logger.info(f"  {label}: TER...")
    res["TER"]   = metric_ter(hyps, refs)

    if run_meteor:
        logger.info(f"  {label}: METEOR...")
        res["METEOR"] = metric_meteor(hyps, refs)

    if run_bertscore:
        logger.info(f"  {label}: BERTScore...")
        bp, br, bf = metric_bertscore(hyps, refs)
        res["BERTScore_P"] = bp
        res["BERTScore_R"] = br
        res["BERTScore_F"] = bf

    if run_comet and srcs is not None:
        logger.info(f"  {label}: COMET...")
        res["COMET"] = metric_comet(hyps, refs, srcs)

    return res


def _fmt(v):
    if isinstance(v, float):
        return "—" if (v != v) else f"{v:.4f}"
    return str(v)

def pretty_table(rows, key_cols, metric_cols, title):
    all_cols = key_cols + metric_cols
    widths = {c: max(len(c), max(len(_fmt(r.get(c, ""))) for r in rows))
              for c in all_cols}
    sep  = "+" + "+".join("-" * (widths[c] + 2) for c in all_cols) + "+"
    hdr  = "|" + "|".join(f" {c:>{widths[c]}} " for c in all_cols) + "|"
    lines = [f"\n  {title}", sep, hdr, sep]
    for r in rows:
        line = "|" + "|".join(f" {_fmt(r.get(c,'')):>{widths[c]}} "
                               for c in all_cols) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def main():
    pa = argparse.ArgumentParser(
        description="Unified NMT metric evaluator — reads any *predictions*.tsv files",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--pred_dir",
        default=DEFAULT_PRED_DIR,
        help="Folder containing TSV files whose names contain 'predictions'")
    pa.add_argument("--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Output folder for CSVs and tables (default: <pred_dir>/../results)")
    pa.add_argument("--src_dir",
        default=DEFAULT_SRC_DIR,
        help="Folder containing <split>.sa source file (for COMET)")
    pa.add_argument("--src_file",
        default=DEFAULT_SRC_FILE,
        help="Direct path to a .sa source file — overrides --src_dir + --split")
    pa.add_argument("--split",
        default=DEFAULT_SPLIT,
        help="Split name used with --src_dir to locate <split>.sa (default: test)")
    pa.add_argument("--no_bertscore", action="store_true")
    pa.add_argument("--no_meteor",    action="store_true")
    pa.add_argument("--no_comet",     action="store_true")
    args = pa.parse_args()

    pred_dir = args.pred_dir
    out_dir  = args.out_dir or os.path.join(
        os.path.dirname(pred_dir.rstrip("/\\")), "results")
    os.makedirs(out_dir, exist_ok=True)

    run_bertscore = (not args.no_bertscore) and BERTSCORE_OK
    run_meteor    = (not args.no_meteor)    and METEOR_OK
    run_comet     = (not args.no_comet)     and COMET_OK

    resolved_src = resolve_source_file(args.src_file, args.src_dir, args.split)
    srcs_all: Optional[List[str]] = None

    if run_comet:
        if resolved_src:
            srcs_all = read_source_lines(resolved_src)
        else:
            logger.warning(
                "No source file resolved (provide --src_file or --src_dir) "
                "— COMET will be skipped."
            )
            run_comet = False

    pattern  = os.path.join(pred_dir, "*.tsv")
    all_tsv  = sorted(glob.glob(pattern))
    tsv_files = [f for f in all_tsv if "predictions" in Path(f).name.lower()]

    if not tsv_files:
        logger.error(
            f"No TSV files whose name contains 'predictions' found in: {pred_dir}\n"
            f"  (searched pattern: {pattern})\n"
            f"  Files present: {[Path(f).name for f in all_tsv] or 'none'}"
        )
        sys.exit(1)

    logger.info(f"Found {len(tsv_files)} prediction TSV file(s) in {pred_dir}:")
    for f in tsv_files:
        logger.info(f"  {Path(f).name}")

    base_metrics = ["BLEU", "chrF", "chrF2", "TER"]
    if run_meteor:    base_metrics.append("METEOR")
    if run_bertscore: base_metrics += ["BERTScore_P", "BERTScore_R", "BERTScore_F"]
    if run_comet:     base_metrics.append("COMET")

    overall_rows: List[Dict] = []
    binwise_rows: List[Dict] = []

    for tsv_path in tsv_files:
        tag = Path(tsv_path).stem

        logger.info(f"\n{'='*60}")
        logger.info(f"  Model: {tag}  ({Path(tsv_path).name})")
        logger.info(f"{'='*60}")

        src_lens, refs, hyps = read_tsv(tsv_path)

        srcs: Optional[List[str]] = None
        if run_comet and srcs_all is not None:
            if len(srcs_all) == len(refs):
                srcs = srcs_all
            elif len(srcs_all) > len(refs):
                logger.warning(f"  Source has more lines ({len(srcs_all)}) than TSV "
                               f"({len(refs)}); truncating source.")
                srcs = srcs_all[:len(refs)]
            else:
                logger.warning(f"  Source has fewer lines ({len(srcs_all)}) than TSV "
                               f"({len(refs)}); COMET skipped for {tag}.")
                srcs = None

        logger.info(f"  Computing overall metrics...")
        overall = compute_all_metrics(
            hyps, refs, srcs,
            run_bertscore=run_bertscore,
            run_meteor=run_meteor,
            run_comet=(run_comet and srcs is not None),
            label=f"{tag} overall",
        )
        row = {"Model": tag, "N": len(refs)}
        row.update(overall)
        overall_rows.append(row)

        logger.info(f"\n  [{tag}] Overall results ({len(refs):,} sentences):")
        for k, v in overall.items():
            logger.info(f"    {k:<18} {_fmt(v)}")

        logger.info(f"\n  Computing bin-wise metrics...")
        buckets = [assign_bucket(n) for n in src_lens]

        for b in range(1, 6):
            lbl = BUCKET_LABELS[b]
            idx = [i for i, bk in enumerate(buckets) if bk == b]
            if not idx:
                logger.info(f"    {lbl}: no samples — skipping")
                continue

            bh = [hyps[i] for i in idx]
            br = [refs[i]  for i in idx]
            bs = [srcs[i]  for i in idx] if srcs else None

            logger.info(f"    {lbl}: {len(idx)} samples...")
            bm = compute_all_metrics(
                bh, br, bs,
                run_bertscore=run_bertscore,
                run_meteor=run_meteor,
                run_comet=(run_comet and bs is not None),
                label=f"{tag}/{lbl}",
            )
            brow = {"Model": tag, "Bucket": lbl, "N": len(idx)}
            brow.update(bm)
            binwise_rows.append(brow)

    if not overall_rows:
        logger.error("No results computed. Exiting.")
        sys.exit(1)

    overall_key_cols = ["Model", "N"]
    binwise_key_cols = ["Model", "Bucket", "N"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def write_csv(rows, key_cols, path):
        if not rows: return
        metric_cols = [c for c in rows[0] if c not in key_cols]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=key_cols + metric_cols)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: _fmt(r.get(k, "")) for k in key_cols + metric_cols})
        logger.info(f"Saved → {path}")

    overall_csv = os.path.join(out_dir, f"overall_metrics_{ts}.csv")
    binwise_csv = os.path.join(out_dir, f"binwise_metrics_{ts}.csv")
    write_csv(overall_rows, overall_key_cols, overall_csv)
    write_csv(binwise_rows, binwise_key_cols, binwise_csv)

    json_path = os.path.join(out_dir, f"all_metrics_{ts}.json")
    json.dump({"timestamp": ts, "overall": overall_rows, "binwise": binwise_rows},
              open(json_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved → {json_path}")

    metric_cols = base_metrics

    overall_table = pretty_table(
        overall_rows, overall_key_cols, metric_cols,
        "OVERALL METRICS — all models")

    binwise_table = pretty_table(
        binwise_rows, binwise_key_cols, metric_cols,
        "BIN-WISE METRICS — all models × buckets")

    print("\n" + "=" * 80)
    print(overall_table)
    print("\n" + "=" * 80)
    print(binwise_table)
    print("\n" + "=" * 80)

    for table, name in [(overall_table, f"overall_metrics_{ts}.txt"),
                        (binwise_table, f"binwise_metrics_{ts}.txt")]:
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        logger.info(f"Saved → {p}")

    logger.info("\nDone. All outputs written to: " + out_dir)


if __name__ == "__main__":
    main()
