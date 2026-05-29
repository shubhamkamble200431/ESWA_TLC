import os
import sys
import logging
import argparse

import torch

import sys as _sys
import types as _types
import importlib.util as _ilu
import os as _os

_os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
_os.environ.pop("DEEPSPEED_CONFIG_FILE", None)

if "deepspeed" not in _sys.modules:
    _ds_name = "deepspeed"
    _ds = _types.ModuleType(_ds_name)
    _ds.__spec__ = _ilu.spec_from_loader(_ds_name, loader=None)
    _ds.__spec__.submodule_search_locations = []
    _ds.DeepSpeedEngine = None
    for _sub in ["ops", "ops.op_builder", "git_version_info",
                 "runtime", "runtime.zero", "utils"]:
        _full = f"deepspeed.{_sub}"
        _m = _types.ModuleType(_full)
        _m.__spec__ = _ilu.spec_from_loader(_full, loader=None)
        _sys.modules[_full] = _m
    _sys.modules[_ds_name] = _ds

from configs.combo_configs import get_combo_config, ALL_COMBO_CONFIGS, COMBO_LABELS

def get_config(case_id: str):
    return get_combo_config(case_id)

ALL_CASES = list(ALL_COMBO_CONFIGS.keys())

from data.dataset import (
    load_parallel_corpus,
    SANHINDataset,
    concat_augment,
    load_bt_data,
)
from data.transliterate import transliterate_corpus
from models.model_builder import build_model_and_tokenizer
from training.trainer import SANHINTrainer, build_training_args
from transformers import DataCollatorForSeq2Seq
from training.curriculum import CurriculumScheduler
from evaluation.metrics import make_compute_metrics
from evaluation.evaluate import evaluate_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train(case_id: str, overrides: dict = None):
    cfg = get_config(case_id)
    from evaluation import metrics as _metrics_mod
    _metrics_mod.COMET_EVAL_EVERY = getattr(cfg, "comet_eval_every", 5)
    _metrics_mod._eval_call_count = 0
    logger.info(f"[Metrics] COMET will run every {_metrics_mod.COMET_EVAL_EVERY} epochs.")

    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                logger.info(f"  Config override: {k} = {v}")

    import os as _os
    _os.environ.pop("DEEPSPEED_CONFIG_FILE", None)
    _os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"

    logger.info(f"\n{'='*60}")
    logger.info(f"  Experiment: {cfg.experiment_name}")
    logger.info(f"  Case:       {cfg.case_id}")
    logger.info(f"  Model:      {cfg.base_model}")
    logger.info(f"  {cfg.description}")
    logger.info(f"{'='*60}\n")

    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("Loading training data...")
    train_src, train_tgt = load_parallel_corpus(
        cfg.data_dir, cfg.train_src, cfg.train_tgt,
        max_samples=cfg.max_train_samples,
    )
    val_src, val_tgt = load_parallel_corpus(
        cfg.data_dir, cfg.val_src, cfg.val_tgt,
        max_samples=cfg.max_val_samples,
    )

    if cfg.transliterate_to_devanagari:
        logger.info("Transliterating Devanagari normalisation...")
        train_src = transliterate_corpus(train_src)
        val_src   = transliterate_corpus(val_src)

    if cfg.use_concat_augmentation:
        logger.info("Applying sentence concatenation augmentation (B4-B5)...")
        train_src, train_tgt = concat_augment(
            train_src, train_tgt,
            ratio=cfg.concat_aug_ratio,
            boundaries=cfg.bucket_boundaries,
            seed=cfg.seed,
        )

    if cfg.use_back_translation and cfg.bt_data_path:
        logger.info("Loading back-translation augmented data...")
        bt_src, bt_tgt = load_bt_data(cfg.bt_data_path)
        if bt_src:
            train_src = train_src + bt_src
            train_tgt = train_tgt + bt_tgt
            logger.info(
                f"Added {len(bt_src):,} BT pairs. "
                f"Total training: {len(train_src):,}"
            )

    logger.info(
        f"Dataset sizes — train: {len(train_src):,}  val: {len(val_src):,}"
    )

    logger.info("Building model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer(cfg)

    val_dataset = SANHINDataset(
        src_sentences=val_src,
        tgt_sentences=val_tgt,
        tokenizer=tokenizer,
        src_lang=cfg.src_lang,
        tgt_lang=cfg.tgt_lang,
        max_input_length=cfg.max_input_length,
        max_target_length=cfg.max_target_length,
        use_length_token=cfg.use_length_token,
        bucket_boundaries=cfg.bucket_boundaries,
    )

    training_args = build_training_args(cfg)

    compute_metrics = make_compute_metrics(tokenizer, sources=val_src)

    _eval_base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    def eval_collator(features):
        for f in features:
            f.pop("bucket", None)
        return _eval_base_collator(features)

    logger.info("Initializing trainer...")
    trainer = SANHINTrainer(
        cfg=cfg,
        tokenizer=tokenizer,
        full_train_src=train_src,
        full_train_tgt=train_tgt,
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        data_collator=eval_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    best_path = os.path.join(cfg.output_dir, "checkpoint-best")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    logger.info(f"Best model saved to: {best_path}")

    logger.info("Running final test evaluation...")
    test_metrics = evaluate_checkpoint(
        checkpoint_path=best_path,
        case_id=case_id,
        split="test",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    logger.info(f"\nExperiment {case_id} complete.")
    logger.info(f"  Overall BLEU: {test_metrics.get('bleu', 0):.2f}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="SAN→HIN NMT Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --case_id X-0000
  python main.py --case_id X-1100 --epochs 15 --lr 3e-5
  python main.py --case_id X-1111 --eval_only --checkpoint outputs/X-1111/checkpoint-best
  python main.py --run_all
  python main.py --list_configs
        """
    )

    parser.add_argument(
        "--case_id", type=str, default=None,
        choices=ALL_CASES,
        help="Case ID: X-0000 to X-1111",
    )
    parser.add_argument(
        "--run_all", action="store_true",
        help="Run all combinatorial cases sequentially",
    )
    parser.add_argument(
        "--list_configs", action="store_true",
        help="Print all available configs and exit",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training; only evaluate a checkpoint",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint path for --eval_only",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Data split for evaluation",
    )

    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--data_dir",   type=str,   default=None)
    parser.add_argument("--output_dir", type=str,   default=None)
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--fp16",       action="store_true", default=None)

    args = parser.parse_args()

    if args.list_configs:
        print("\nCombinatorial Grid (X-BLRA  B=BBS L=LTC R=RoPE A=ALiBi):")
        print("=" * 70)
        for cid, label in COMBO_LABELS.items():
            fl = cid[2:]
            b = "Y" if fl[0]=="1" else "N"
            l = "Y" if fl[1]=="1" else "N"
            r = "Y" if fl[2]=="1" else "N"
            a = "Y" if fl[3]=="1" else "N"
            print(f"  {cid}  {label:<30}  BBS={b}  LTC={l}  RoPE={r}  ALiBi={a}")
        return

    if args.eval_only:
        if not args.case_id:
            parser.error("--case_id required with --eval_only")
        if not args.checkpoint:
            cfg = get_config(args.case_id)
            args.checkpoint = os.path.join(cfg.output_dir, "checkpoint-best")
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            case_id=args.case_id,
            split=args.split,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return

    overrides = {}
    if args.epochs:     overrides["num_train_epochs"] = args.epochs
    if args.batch_size: overrides["per_device_train_batch_size"] = args.batch_size
    if args.lr:         overrides["learning_rate"] = args.lr
    if args.data_dir:   overrides["data_dir"] = args.data_dir
    if args.output_dir: overrides["output_dir"] = args.output_dir
    if args.seed:       overrides["seed"] = args.seed
    if args.fp16:       overrides["fp16"] = True

    if args.run_all:
        all_results = {}
        for cid in ALL_COMBO_CONFIGS.keys():
            logger.info(f"\n{'#'*60}\n  Starting {cid}\n{'#'*60}")
            try:
                metrics = train(cid, overrides=overrides)
                all_results[cid] = metrics
            except Exception as e:
                logger.error(f"Case {cid} failed: {e}", exc_info=True)
                all_results[cid] = {"error": str(e)}

        print("\n" + "=" * 70)
        print("  SUMMARY — All Experiments")
        print("=" * 70)
        print(f"  {'Case':<12} {'BLEU':>8}  {'chrF':>8}")
        print("-" * 70)
        for cid, m in all_results.items():
            if "error" in m:
                print(f"  {cid:<12}  ERROR: {m['error'][:40]}")
            else:
                print(
                    f"  {cid:<12} "
                    f"{m.get('bleu', 0):>8.2f}  "
                    f"{m.get('chrf', 0):>8.2f}"
                )
        return

    if not args.case_id:
        parser.error("Provide --case_id, --run_all, --list_configs, or --eval_only")

    train(args.case_id, overrides=overrides)


if __name__ == "__main__":
    main()
