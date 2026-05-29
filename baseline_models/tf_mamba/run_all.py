"""
run_all.py — Master orchestration script.

Handles ALL model families in one place:
  • Vanilla Transformer (6 / 12 / 18 layers)   — trained from scratch
  • Mamba (small / base / large)                — trained from scratch
  • M2M100 (418M / 1.2B)                        — fine-tuned from HuggingFace
  • LongT5 (base / large)                       — fine-tuned from HuggingFace
  • mT5    (small / base / large)               — fine-tuned from HuggingFace

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Single model (works for both scratch and pretrained)
  python run_all.py --model transformer_6
  python run_all.py --model m2m100_418m
  python run_all.py --model mt5_base

  # Comma-separated queue
  python run_all.py --model transformer_6,mt5_small,m2m100_418m

  # Named groups
  python run_all.py --only transformer     # transformer_6/12/18
  python run_all.py --only mamba           # mamba small/base/large
  python run_all.py --only m2m100          # m2m100 418m + 1.2b
  python run_all.py --only longt5          # longt5 base + large
  python run_all.py --only mt5             # mt5 small/base/large
  python run_all.py --only pretrained      # all pretrained models
  python run_all.py --only scratch         # all scratch models
  python run_all.py --only all             # everything (default)

  # Mix groups and singles
  python run_all.py --only transformer --model mt5_small

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --skip_training   skip training/fine-tuning; jump to eval + plots
  --skip_eval       skip test-set evaluation
  --skip_plots      skip visualisation
  --eval_only       alias for --skip_training
  --train_only      alias for --skip_eval --skip_plots
"""

import argparse
import os
import time

from config import (
    ALL_CONFIGS, ALL_SCRATCH_CONFIGS, ALL_PRETRAINED_CONFIGS,
    TRANSFORMER_CONFIGS, MAMBA_CONFIGS, PRETRAINED_CONFIGS,
    OUTPUT_DIR, BIN_NAMES,
)
from dataset  import train_bpe
from train    import train_model
from predict  import evaluate_model
from finetune import finetune_model
from predict_pretrained import evaluate_pretrained
from visualize import main as visualize_main
from utils import setup_logging, save_results

logger = setup_logging("run_all")

# ── Group aliases ────────────────────────────────────────────────
GROUP_MAP = {
    "transformer": list(TRANSFORMER_CONFIGS.keys()),
    "mamba":       list(MAMBA_CONFIGS.keys()),
    "m2m100":      [k for k, v in PRETRAINED_CONFIGS.items() if v.model_type == "m2m100"],
    "longt5":      [k for k, v in PRETRAINED_CONFIGS.items() if v.model_type == "longt5"],
    "mt5":         [k for k, v in PRETRAINED_CONFIGS.items() if v.model_type == "mt5"],
    "pretrained":  list(PRETRAINED_CONFIGS.keys()),
    "scratch":     list(ALL_SCRATCH_CONFIGS.keys()),
    "all":         list(ALL_CONFIGS.keys()),
}


# ─────────────────────────────────────────────────────────────────
# Config resolution
# ─────────────────────────────────────────────────────────────────

def resolve_configs(only: str | None, model: str | None) -> list[str]:
    selected = []

    if only:
        group = GROUP_MAP.get(only)
        if group is None:
            raise ValueError(f"Unknown group '{only}'. Choose from: {list(GROUP_MAP)}")
        selected.extend(group)

    if model:
        for name in model.split(","):
            name = name.strip()
            if name not in ALL_CONFIGS:
                raise ValueError(
                    f"Unknown model '{name}'.\nAvailable: {list(ALL_CONFIGS)}"
                )
            if name not in selected:
                selected.append(name)

    if not selected:
        selected = list(ALL_CONFIGS.keys())

    return selected


# ─────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────

def run_pipeline(
    configs_to_run: list[str],
    skip_training: bool = False,
    skip_eval:     bool = False,
    skip_plots:    bool = False,
):
    os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)

    # split into scratch vs pretrained
    scratch_models    = [n for n in configs_to_run if n in ALL_SCRATCH_CONFIGS]
    pretrained_models = [n for n in configs_to_run if n in ALL_PRETRAINED_CONFIGS]

    _banner(f"Pipeline  —  {len(scratch_models)} scratch + "
            f"{len(pretrained_models)} pretrained  =  {len(configs_to_run)} total")
    if scratch_models:
        logger.info("  Scratch    : %s", scratch_models)
    if pretrained_models:
        logger.info("  Pretrained : %s", pretrained_models)

    # ── Step 0: BPE (scratch only) ──────────
    if scratch_models:
        _banner("STEP 0 — BPE tokeniser (scratch models)")
        train_bpe()

    # ── Step 1: Training ────────────────────
    train_results = {}
    if not skip_training:
        # ── 1a: scratch ──
        if scratch_models:
            _banner(f"STEP 1a — Training scratch models ({len(scratch_models)})")
            for i, name in enumerate(scratch_models, 1):
                logger.info("\n[%d/%d] ▶ Training: %s", i, len(scratch_models), name)
                t0      = time.time()
                bleu    = train_model(name)
                elapsed = time.time() - t0
                train_results[name] = {"val_bleu": round(bleu, 2),
                                       "time_minutes": round(elapsed / 60, 1)}
                logger.info("  ✓ %.1f min  |  val BLEU = %.2f", elapsed / 60, bleu)

        # ── 1b: pretrained ──
        if pretrained_models:
            _banner(f"STEP 1b — Fine-tuning pretrained models ({len(pretrained_models)})")
            for i, name in enumerate(pretrained_models, 1):
                logger.info("\n[%d/%d] ▶ Fine-tuning: %s", i, len(pretrained_models), name)
                t0      = time.time()
                bleu    = finetune_model(name)
                elapsed = time.time() - t0
                train_results[name] = {"val_bleu": round(bleu, 2),
                                       "time_minutes": round(elapsed / 60, 1)}
                logger.info("  ✓ %.1f min  |  val BLEU = %.2f", elapsed / 60, bleu)
    else:
        logger.info("STEP 1 — Skipped (--skip_training / --eval_only)")

    # ── Step 2: Evaluation ──────────────────
    eval_results = {}
    if not skip_eval:
        _banner(f"STEP 2 — Test-set evaluation ({len(configs_to_run)} models)")

        for i, name in enumerate(scratch_models, 1):
            logger.info("\n[%d/%d] ▶ Evaluating scratch: %s",
                        i, len(scratch_models), name)
            res = evaluate_model(name)
            if res:
                eval_results[name] = res

        for i, name in enumerate(pretrained_models, 1):
            logger.info("\n[%d/%d] ▶ Evaluating pretrained: %s",
                        i, len(pretrained_models), name)
            res = evaluate_pretrained(name)
            if res:
                eval_results[name] = res
    else:
        logger.info("STEP 2 — Skipped (--skip_eval / --train_only)")

    # ── Step 3: Visualisation ───────────────
    if not skip_plots:
        _banner("STEP 3 — Visualisation")
        visualize_main()
    else:
        logger.info("STEP 3 — Skipped (--skip_plots / --train_only)")

    # ── Summary ─────────────────────────────
    if eval_results:
        print_summary(eval_results)

    report = {
        "configs_run":   configs_to_run,
        "scratch":       scratch_models,
        "pretrained":    pretrained_models,
        "train_summary": train_results,
        "test_results":  eval_results,
    }
    report_path = os.path.join(OUTPUT_DIR, "results", "full_report.json")
    save_results(report, report_path)
    logger.info("Full report → %s", report_path)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _banner(msg: str):
    logger.info("\n" + "="*64)
    logger.info("  %s", msg)
    logger.info("="*64)


def print_summary(eval_results: dict):
    sep     = "=" * 95
    bin_hdr = "  ".join(f"{b:>6}" for b in BIN_NAMES)
    print(f"\n{sep}")
    print("  FINAL RESULTS SUMMARY")
    print(sep)
    print(f"  {'Model':<22} {'Type':<12} {'BLEU':>6} {'chrF':>6} {'TER':>6}  |  {bin_hdr}")
    print("-" * 95)
    for name in list(ALL_CONFIGS.keys()):
        if name not in eval_results:
            continue
        mtype = ALL_CONFIGS[name].model_type
        ov    = eval_results[name]["overall"]
        bv    = "  ".join(f"{eval_results[name]['bins'][b]['bleu']:>6.2f}" for b in BIN_NAMES)
        print(f"  {name:<22} {mtype:<12} {ov['bleu']:>6.2f} {ov['chrf']:>6.2f} {ov['ter']:>6.2f}  |  {bv}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sanskrit→Hindi NMT — full pipeline (scratch + pretrained)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sel = p.add_argument_group("Model selection")
    sel.add_argument(
        "--model", metavar="NAME[,NAME,...]", default=None,
        help="One or more model names comma-separated",
    )
    sel.add_argument(
        "--only", choices=list(GROUP_MAP.keys()), default=None,
        metavar="{transformer,mamba,m2m100,longt5,mt5,pretrained,scratch,all}",
        help="Run a named group (combinable with --model)",
    )
    pipe = p.add_argument_group("Pipeline control")
    pipe.add_argument("--skip_training", action="store_true")
    pipe.add_argument("--skip_eval",     action="store_true")
    pipe.add_argument("--skip_plots",    action="store_true")
    pipe.add_argument("--eval_only",  action="store_true",
                      help="Alias for --skip_training")
    pipe.add_argument("--train_only", action="store_true",
                      help="Alias for --skip_eval --skip_plots")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    skip_training = args.skip_training or args.eval_only
    skip_eval     = args.skip_eval     or args.train_only
    skip_plots    = args.skip_plots    or args.train_only

    try:
        configs = resolve_configs(args.only, args.model)
    except ValueError as e:
        print(f"\n[ERROR] {e}\n")
        raise SystemExit(1)

    logger.info("Models queued  : %s", configs)
    logger.info("skip_training  : %s", skip_training)
    logger.info("skip_eval      : %s", skip_eval)
    logger.info("skip_plots     : %s", skip_plots)

    run_pipeline(configs, skip_training=skip_training,
                 skip_eval=skip_eval, skip_plots=skip_plots)