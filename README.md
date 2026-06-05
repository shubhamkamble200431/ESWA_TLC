# Learning the Decoder When to Stop: Improving Long-Sequence Neural Machine Translation via Token-Length Conditioning in Low-Resource Indic Languages

Fine-tuning NLLB-200-distilled-600M for Sanskrit (Devanagari) → Hindi translation with uniform quality across sentence-length buckets.

Fine-tuning with bucket-stratified evaluation across 5 sentence-length buckets (B1–B5).

## Repository Structure

```
ESWA_TLC/
├── data/                     # Parallel corpus (train/val/test .sa/.hi)
├── NLLB_configs/             # Main NLLB-600M experiment pipeline
│   ├── main.py               # Run a single X-* experiment
│   ├── run_experiments.py    # Sequential multi-experiment runner
│   ├── standalone_eval.py    # Evaluate checkpoints without training code
│   ├── configs/              # Config dataclasses (base + 16 combo configs)
│   ├── data/                 # Dataset, sampler, augmentation
│   ├── models/               # Model builder, RoPE, ALiBi, hierarchical encoder
│   ├── training/             # Trainer, curriculum, loss
│   ├── evaluation/           # Metrics (BLEU/chrF/COMET), evaluate, plots
│   └── scripts/              # Back-translation utility
├── baseline_models/          # mBART50, mT5, IndicTrans2, NLLB-200M, Mamba
├── calculate_metrics/        # Post-hoc evaluation on saved prediction TSVs
├── predictions/              # Saved prediction TSV files
├── utils/                    # Data preparation utilities
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.1+.

## Data

Parallel text files: one sentence per line.

| File | Role | Lines |
|------|------|-------|
| `data/train.sa` / `train.hi` | Training | 58,106 |
| `data/val.sa` / `val.hi` | Validation | ~7,000 |
| `data/test.sa` / `test.hi` | Test (raw) | 7,264 |
| `data/test_7264_5.sa` / `test_7264_5.hi` | Test (resampled, equal bucket sizes) | 7,260 |

Prepare `test_7264_5.*` files from the raw test set:

```bash
python utils/resample_7264_5.py \
    --src data/test.sa --tgt data/test.hi \
    --out_src data/test_7264_5.sa --out_tgt data/test_7264_5.hi \
    --plot data/resample_plot.png
```

## Experiments (NLLB_configs)

### Combinatorial Ablation Grid

16 configs: `X-{BBS}{LTC}{RoPE}{ALiBi}` where each flag is 0 (off) or 1 (on).

| Flag | Technique | Description |
|------|-----------|-------------|
| BBS | Bucket-Balanced Sampling | WeightedRandomSampler oversampling B4-B5 |
| LTC | Length Token Conditioning | Prepend `<LEN_BUCKET_X>` to decoder input |
| RoPE | Rotary Position Embedding | Replace learned absolute PE |
| ALiBi | Attention with Linear Biases | Additive position bias on attention |

### Run a single experiment

```bash
cd NLLB_configs
python main.py --case_id X-1111
python main.py --case_id X-0000 --epochs 20 --batch_size 16 --lr 5e-5
python main.py --case_id X-1111 --eval_only --checkpoint outputs/X-1111/checkpoint-best
python main.py --list_configs
```

### Run multiple experiments sequentially

```bash
cd NLLB_configs
python run_experiments.py
python run_experiments.py --cases X-0000 X-1000 X-1111
python run_experiments.py --cases all --skip_done --dry_run
```

### Evaluate a saved checkpoint

```bash
cd NLLB_configs
python standalone_eval.py --cases X-1111
python standalone_eval.py --cases all --data_dir ../data --split test_7264_5
```

### Back-translation (optional augmentation)

```bash
cd NLLB_configs
python scripts/back_translate.py --hi_mono_file ../data/hindi_mono.txt --out_dir ../data/bt_augmented
```

## Baseline Models

Each baseline has `train.py` and `inference_and_eval.py`:

| Baseline | Directory | Model |
|----------|-----------|-------|
| mBART50 | `baseline_models/mBART50_many_to_many/` | `facebook/mbart-large-50-many-to-many-mmt` |
| mT5 | `baseline_models/mT5/` | `google/mt5-large` |
| IndicTrans2 | `baseline_models/indictrans2/` | AI4Bharat IndicTrans2 |
| NLLB-200M | `baseline_models/nllb_200M_distilled/` | `facebook/nllb-200-distilled-600M` |

```bash
python baseline_models/mBART50_many_to_many/train.py --data_dir data --output_dir outputs/mBART50 --epochs 20
python baseline_models/mBART50_many_to_many/inference_and_eval.py --checkpoint outputs/mBART50/checkpoint-best --data_dir data

python baseline_models/mT5/train.py --data_dir data --output_dir outputs/mT5
python baseline_models/mT5/inference_and_eval.py --checkpoint outputs/mT5/checkpoint-best --data_dir data
```

## Utils

```bash
# Convert .sa/.hi text files to HuggingFace JSON format
python utils/text2json.py --src data/train.sa --tgt data/train.hi --out data/train.json

# Convert prediction .txt files to linked .tsv (with source + reference columns)
python utils/txt_to_tsv.py --pred_dir predictions/ --src_file data/test_7264_5.sa --ref_file data/test_7264_5.hi

# Resample test set to equal 1452 sentences per bucket
python utils/resample_7264_5.py --src data/test.sa --tgt data/test.hi \
    --out_src data/test_7264_5.sa --out_tgt data/test_7264_5.hi
```

## Metrics

| Metric | Description | Target |
|----------|-------------|----------|
| BLEU | corpus_bleu (FLORES-200 tokenizer) | Higher is better |
| chrF2 | Character n-gram F-score with word order information | Higher is better |
| TER | Translation Edit Rate; measures the number of edits required to transform a hypothesis into the reference translation | Lower is better |
| METEOR | Alignment-based metric incorporating exact, stem, synonym, and paraphrase matches | Higher is better |
| COMET | Neural MT evaluation metric (wmt22-comet-da) | Higher is better |
| BERTScore-F1 (BSF) | Semantic similarity measured using contextual embeddings and reported as the BERTScore F1 value | Higher is better |