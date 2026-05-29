import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.combo_configs import (
    get_combo_config, ALL_COMBO_CONFIGS, COMBO_LABELS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_experiments")

COMBO_CASE_IDS = list(ALL_COMBO_CONFIGS.keys())
ALL_CASE_IDS   = COMBO_CASE_IDS


def _get_label(case_id: str) -> str:
    return f"{case_id}  [{COMBO_LABELS.get(case_id, '')}]"


def _get_config(case_id: str):
    return get_combo_config(case_id)


def _checkpoint_exists(case_id: str, checkpoint_root: str = "outputs") -> bool:
    cfg  = _get_config(case_id)
    best = os.path.join(cfg.output_dir, "checkpoint-best")
    return os.path.isdir(best)


def _print_case_table():
    print("\n" + "─" * 70)
    print("  COMBINATORIAL GRID  (X-BLRA  where B=BBS L=LTC R=RoPE A=ALiBi)")
    print("─" * 70)
    for cid, label in COMBO_LABELS.items():
        flags = cid[2:]
        desc  = f"BBS={'✓' if flags[0]=='1' else '✗'}  LTC={'✓' if flags[1]=='1' else '✗'}  " \
                f"RoPE={'✓' if flags[2]=='1' else '✗'}  ALiBi={'✓' if flags[3]=='1' else '✗'}"
        print(f"  {cid:<8}  {label:<28}  {desc}")
    print()


def prompt_interactive() -> List[str]:
    _print_case_table()

    print("─" * 70)
    print("  SELECT CASES TO TRAIN")
    print("─" * 70)
    print("  Presets:")
    print("    all        — all X-0000…X-1111 cases")
    print("    combo      — X-0000 through X-1111")
    print()
    print("  Or list case IDs separated by spaces:")
    print("    e.g.  X-0001 X-1001 X-1111")
    print()

    raw = input("  Your selection [all]: ").strip()
    if not raw:
        raw = "all"

    chosen = _resolve_preset(raw.lower())
    if chosen is None:
        tokens = raw.upper().split()
        chosen = []
        for t in tokens:
            if t.startswith("X-"):
                chosen.append(t)
        bad = [c for c in chosen if c not in ALL_CASE_IDS]
        if bad:
            print(f"  [WARN] Unknown IDs ignored: {bad}")
            chosen = [c for c in chosen if c in ALL_CASE_IDS]
        if not chosen:
            print("  No valid cases. Exiting."); sys.exit(0)

    print(f"\n  Selected {len(chosen)} cases: {' '.join(chosen)}")

    print()
    print("  TRAINING ORDER")
    print("  ──────────────")
    print("  1. As listed above (default)")
    print("  2. Custom order — type space-separated IDs")
    print()
    order_raw = input("  Order [1]: ").strip() or "1"

    if order_raw == "2":
        custom_raw = input("  Enter custom order: ").strip().upper().split()
        custom = [t for t in custom_raw if t.startswith("X-")]
        chosen = [c for c in custom if c in chosen]
        if not chosen:
            print("  No valid custom order. Using original selection.")
            chosen = [c for c in ALL_CASE_IDS if c in chosen]

    print()
    print("  HYPERPARAMETER OVERRIDES  (press Enter to keep defaults)")
    print()
    overrides = {}
    for name, key, cast in [
        ("Epochs      [20]", "num_train_epochs", int),
        ("Batch size  [16]", "per_device_train_batch_size", int),
        ("Learning rate [5e-5]", "learning_rate", float),
        ("Seed        [42]", "seed", int),
    ]:
        val = input(f"    {name}: ").strip()
        if val:
            try:
                overrides[key] = cast(val)
            except ValueError:
                print(f"    [WARN] Invalid value '{val}' — keeping default.")

    print()
    skip_raw = input("  Skip cases with existing checkpoint-best? [y/N]: ").strip().lower()
    skip_done = skip_raw in ("y", "yes")

    return chosen, overrides, skip_done


def _resolve_preset(raw: str) -> Optional[List[str]]:
    if raw == "all":
        return ALL_CASE_IDS[:]
    if raw == "combo":
        return COMBO_CASE_IDS[:]
    return None


def train_case(case_id: str, overrides: dict, log_path: str) -> dict:
    import torch

    logger.info(f"\n{'#'*60}\n  Training: {_get_label(case_id)}\n{'#'*60}")

    try:
        from main import train
        metrics = train(case_id, overrides=overrides)
        result  = {"case_id": case_id, "status": "done", "metrics": metrics,
                   "timestamp": datetime.now().isoformat()}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[{case_id}] FAILED:\n{tb}")
        result = {"case_id": case_id, "status": "error",
                  "error": str(e), "traceback": tb,
                  "timestamp": datetime.now().isoformat()}

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return result


def print_run_summary(results: Dict[str, dict]):
    W = 90
    print("\n" + "=" * W)
    print("  TRAINING RUN SUMMARY")
    print("=" * W)
    print(f"  {'Case':<10} {'Status':<8} {'BLEU':>7} {'COMET':>7}  Label")
    print("-" * W)
    for cid, r in results.items():
        if r.get("status") == "done":
            m = r.get("metrics", {})
            print(
                f"  {cid:<10} {'✓ done':<8} "
                f"{m.get('bleu', 0):7.2f} "
                f"{m.get('comet', 0):7.3f}  "
                f"{_get_label(cid)}"
            )
        elif r.get("status") == "skipped":
            print(f"  {cid:<10} {'⏭ skip':<8}  (checkpoint already exists)")
        else:
            err = str(r.get("error", ""))[:50]
            print(f"  {cid:<10} {'✗ error':<8}  {err}")
    print("=" * W + "\n")


def main():
    pa = argparse.ArgumentParser(
        description="Sequential training runner — SAN→HIN NMT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets for --cases:
  all        all 16 combinatorial cases
  combo      X-0000 through X-1111

Examples:
  python run_experiments.py
  python run_experiments.py --cases all --skip_done
  python run_experiments.py --cases X-0001 X-1001 X-1111
  python run_experiments.py --cases combo --epochs 10
  python run_experiments.py --cases all --dry_run
        """)

    pa.add_argument("--cases", nargs="*", default=None,
        help="Case IDs or preset (all/combo). Omit for interactive.")
    pa.add_argument("--epochs",     type=int,   default=None)
    pa.add_argument("--batch_size", type=int,   default=None)
    pa.add_argument("--lr",         type=float, default=None)
    pa.add_argument("--seed",       type=int,   default=None)
    pa.add_argument("--skip_done",  action="store_true",
        help="Skip cases whose checkpoint-best already exists.")
    pa.add_argument("--dry_run",    action="store_true",
        help="Print the training plan without executing.")
    pa.add_argument("--log_dir",    default="runs",
        help="Directory for the rolling training log (default: runs/).")
    pa.add_argument("--list",       action="store_true",
        help="Print all available case IDs and exit.")

    args = pa.parse_args()

    if args.list:
        _print_case_table()
        return

    overrides = {}
    skip_done = args.skip_done

    if args.cases is None:
        chosen, overrides, skip_done = prompt_interactive()
    else:
        raw_cases = args.cases
        if len(raw_cases) == 1:
            resolved = _resolve_preset(raw_cases[0].lower())
            chosen   = resolved if resolved is not None else raw_cases
        else:
            chosen = raw_cases

        chosen = [c.upper() if c.upper().startswith("X-") else c for c in chosen]
        bad = [c for c in chosen if c not in ALL_CASE_IDS]
        if bad:
            pa.error(f"Unknown case IDs: {bad}\nValid: {ALL_CASE_IDS}")

        if args.epochs:     overrides["num_train_epochs"]             = args.epochs
        if args.batch_size: overrides["per_device_train_batch_size"]  = args.batch_size
        if args.lr:         overrides["learning_rate"]                = args.lr
        if args.seed:       overrides["seed"]                         = args.seed

    print(f"\n{'='*65}")
    print(f"  TRAINING PLAN — {len(chosen)} cases")
    print(f"{'='*65}")
    for i, cid in enumerate(chosen, 1):
        exists  = _checkpoint_exists(cid)
        will_skip = skip_done and exists
        status  = "⏭ SKIP (done)" if will_skip else ("✓ queued")
        print(f"  {i:>2}. {cid:<10} {status:<18} {_get_label(cid)}")
    if overrides:
        print(f"\n  Overrides: {overrides}")
    print(f"  skip_done = {skip_done}")
    print(f"  dry_run   = {args.dry_run}")
    print(f"{'='*65}\n")

    if args.dry_run:
        print("  Dry run complete — no training executed.")
        return

    os.makedirs(args.log_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"training_log_{ts}.jsonl")
    logger.info(f"Training log: {log_path}")

    all_results  = {}
    total        = len(chosen)
    done = err = skipped = 0

    for i, cid in enumerate(chosen, 1):
        logger.info(f"\n[{i}/{total}] {cid} — {_get_label(cid)}")

        if skip_done and _checkpoint_exists(cid):
            logger.info(f"  checkpoint-best already exists — skipping.")
            all_results[cid] = {"case_id": cid, "status": "skipped"}
            skipped += 1
            continue

        result = train_case(cid, overrides, log_path)
        all_results[cid] = result

        if result.get("status") == "done":
            done += 1
            m = result.get("metrics", {})
            logger.info(
                f"  ✓ done — BLEU={m.get('bleu',0):.2f}  "
                f"COMET={m.get('comet',0):.3f}"
            )
        else:
            err += 1
            logger.error(f"  ✗ error: {result.get('error','')[:80]}")

    print_run_summary(all_results)
    logger.info(f"Run complete — done={done}  error={err}  skipped={skipped}")

    summary_path = os.path.join(args.log_dir, f"summary_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "traceback"}
                for k, v in all_results.items()}
        json.dump(slim, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
