# NLLB_configs

NLLB-200-distilled-600M fine-tuning pipeline for Sanskrit→Hindi NMT with 16 combinatorial ablation configs.

## Entry Points

| Script | Purpose |
|--------|---------|
| `main.py` | Train or evaluate a single X-* config |
| `run_experiments.py` | Run multiple configs sequentially with a menu |
| `standalone_eval.py` | Evaluate saved checkpoints without training code |

## Quick Start

```bash
# Run from NLLB_configs/
python main.py --case_id X-1111
python main.py --case_id X-0000 --eval_only --checkpoint outputs/X-0000/checkpoint-best
python run_experiments.py --cases all --skip_done
python standalone_eval.py --cases X-1000 X-1100 X-1111
```

## Configs (X-BLRA)

| Config | BBS | LTC | RoPE | ALiBi |
|--------|-----|-----|------|-------|
| X-0000 | — | — | — | — |
| X-1000 | ✓ | — | — | — |
| X-0100 | — | ✓ | — | — |
| X-0010 | — | — | ✓ | — |
| X-0001 | — | — | — | ✓ |
| X-1100 | ✓ | ✓ | — | — |
| … | | | | |
| X-1111 | ✓ | ✓ | ✓ | ✓ |

Full list via: `python main.py --list_configs`

## Outputs

Each experiment writes under `outputs/<case_id>/`:
- `checkpoint-best/` — best model by eval BLEU
- `eval_results/` — metrics JSON and example translations
- `plots/` — training dashboard PNGs

## Data Layout

```
../data/
├── train.sa / train.hi
├── val.sa / val.hi
└── test_7264_5.sa / test_7264_5.hi   (resampled equal-bucket test set)
```
