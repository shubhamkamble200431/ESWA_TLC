import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    logger.warning("sacrebleu not installed. Install: pip install sacrebleu")

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

BUCKET_LABELS = {1: "B1_1-5", 2: "B2_6-10", 3: "B3_11-15", 4: "B4_16-20", 5: "B5_20+"}
BUCKET_BOUNDARIES = [5, 10, 15, 20]


def assign_bucket(n_tokens: int) -> int:
    for i, b in enumerate(BUCKET_BOUNDARIES):
        if n_tokens <= b:
            return i + 1
    return 5


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
    tokenizer=None,
    run_comet: bool = True,
) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) != references ({len(references)})"
        )

    if sources:
        buckets = [assign_bucket(len(s.split())) for s in sources]
    else:
        buckets = [assign_bucket(len(r.split())) for r in references]

    results: Dict[str, float] = {}

    if SACREBLEU_AVAILABLE:
        overall_bleu = _compute_bleu(predictions, references)
        overall_chrf = _compute_chrf(predictions, references)
        results["bleu"] = overall_bleu
        results["chrf"] = overall_chrf

        bucket_bleus: Dict[int, float] = {}
        for b in range(1, 6):
            b_preds = [p for p, bk in zip(predictions, buckets) if bk == b]
            b_refs  = [r for r, bk in zip(references,  buckets) if bk == b]
            if not b_preds:
                logger.warning(f"No samples for bucket B{b}.")
                continue
            label = BUCKET_LABELS[b]
            bleu_b = _compute_bleu(b_preds, b_refs)
            chrf_b = _compute_chrf(b_preds, b_refs)
            lr_b   = _compute_length_ratio(b_preds, b_refs)
            results[f"bleu_{label}"] = bleu_b
            results[f"chrf_{label}"] = chrf_b
            results[f"length_ratio_{label}"] = lr_b
            bucket_bleus[b] = bleu_b

        results["length_ratio"] = _compute_length_ratio(predictions, references)

    if COMET_AVAILABLE and sources and run_comet:
        try:
            comet_score = _compute_comet(predictions, references, sources)
            results["comet"] = comet_score
        except Exception as e:
            logger.warning(f"COMET computation failed: {e}")

    return results


def _compute_bleu(predictions: List[str], references: List[str]) -> float:
    if not SACREBLEU_AVAILABLE:
        return 0.0
    result = sacrebleu.corpus_bleu(
        predictions,
        [references],
        tokenize="flores200",
    )
    return result.score


def _compute_chrf(predictions: List[str], references: List[str]) -> float:
    if not SACREBLEU_AVAILABLE:
        return 0.0
    result = sacrebleu.corpus_chrf(
        predictions,
        [references],
        word_order=2,
    )
    return result.score


def _compute_length_ratio(predictions: List[str], references: List[str]) -> float:
    ratios = []
    for p, r in zip(predictions, references):
        r_len = max(len(r.split()), 1)
        ratios.append(len(p.split()) / r_len)
    return float(np.mean(ratios))


_COMET_MODEL_CACHE = None

def _get_comet_model():
    global _COMET_MODEL_CACHE
    if _COMET_MODEL_CACHE is None:
        logger.info("[COMET] Loading model (first call only)...")
        model_path = download_model("Unbabel/wmt22-comet-da")
        _COMET_MODEL_CACHE = load_from_checkpoint(model_path)
        logger.info("[COMET] Model cached.")
    return _COMET_MODEL_CACHE


def _compute_comet(
    predictions: List[str],
    references: List[str],
    sources: List[str],
) -> float:
    comet_model = _get_comet_model()
    data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, references)
    ]
    scores = comet_model.predict(data, batch_size=64, gpus=1)
    return float(np.mean(scores.scores))


COMET_EVAL_EVERY: int = 5
_eval_call_count: int = 0

def _run_comet_this_epoch() -> bool:
    global _eval_call_count
    _eval_call_count += 1
    run = (_eval_call_count % COMET_EVAL_EVERY == 0) or (_eval_call_count == 1)
    if not run:
        logger.info(f"[COMET] Skipping epoch {_eval_call_count} "
                    f"(runs every {COMET_EVAL_EVERY} epochs).")
    return run


def make_compute_metrics(tokenizer, sources: Optional[List[str]] = None):
    def compute_metrics(eval_preds):
        pred_ids, label_ids = eval_preds

        pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        return compute_all_metrics(decoded_preds, decoded_labels, sources=sources, run_comet=_run_comet_this_epoch())

    return compute_metrics
