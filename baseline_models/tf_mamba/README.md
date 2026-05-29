# Sanskrit → Hindi Neural Machine Translation
## Vanilla Transformer (6/12/18 layers) + Mamba Variants

---

## Project Structure

```
sanskrit_hindi_nmt/
├── 65k_data/               ← your data goes here
│   ├── train.sa / train.hi
│   ├── val.sa   / val.hi
│   └── test.sa  / test.hi
│
├── config.py               ← all hyperparameters & model configs
├── dataset.py              ← BPE tokeniser + Dataset / DataLoader
├── model.py                ← VanillaTransformer + MambaSeq2Seq architectures
├── utils.py                ← loss, scheduler, metrics, bin helpers
├── train.py                ← training loop (single or all configs)
├── predict.py              ← inference + bin-wise evaluation on test set
├── visualize.py            ← all plots and visualizations
├── run_all.py              ← master orchestration script
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place your data folder at `sanskrit_hindi_nmt/65k_data/` with files:
`train.sa`, `train.hi`, `val.sa`, `val.hi`, `test.sa`, `test.hi`

---

## Model Configurations

| Config          | Type        | Layers | d_model | Parameters |
|-----------------|-------------|--------|---------|------------|
| transformer_6   | Transformer | 6+6    | 256     | ~18M       |
| transformer_12  | Transformer | 12+12  | 256     | ~33M       |
| transformer_18  | Transformer | 18+18  | 256     | ~47M       |
| mamba_small     | Mamba SSM   | 6      | 256     | ~12M       |
| mamba_base      | Mamba SSM   | 12     | 512     | ~45M       |
| mamba_large     | Mamba SSM   | 18     | 512     | ~65M       |

---

## Usage

### Full pipeline (train → evaluate → visualize everything)
```bash
python run_all.py
```

### Only Transformer variants
```bash
python run_all.py --only transformer
```

### Only Mamba variants
```bash
python run_all.py --only mamba
```

### Skip training (if checkpoints already exist)
```bash
python run_all.py --skip_training
```

### Train a single model
```bash
python train.py --config transformer_6
python train.py --config mamba_base
python train.py --all
```

### Evaluate a single model on test set
```bash
python predict.py --config transformer_6
python predict.py --all
```

### Generate all plots
```bash
python visualize.py
```

---

## Bin-wise Evaluation

Source sentence length bins (based on raw word count in `.sa` files):

| Bin   | Length Range |
|-------|--------------|
| 1-5   | 1–5 words    |
| 6-10  | 6–10 words   |
| 11-15 | 11–15 words  |
| 16-20 | 16–20 words  |
| 20+   | 21+ words    |

Metrics computed per bin: **BLEU**, **chrF**, **TER**

---

## Outputs

```
outputs/
├── bpe.model                          ← shared BPE tokeniser
├── checkpoints/
│   ├── transformer_6_best.pt
│   ├── transformer_12_best.pt
│   └── ...
├── predictions/
│   ├── transformer_6_predictions.tsv  ← src_len | reference | hypothesis
│   └── ...
├── results/
│   ├── transformer_6_test_metrics.json
│   ├── transformer_6_train_history.json
│   ├── full_report.json               ← consolidated report
│   └── ...
├── plots/
│   ├── transformer_6_training_curve.png
│   ├── transformer_6_binwise_metrics.png
│   ├── all_models_overall_comparison.png
│   ├── all_models_binwise_heatmap.png
│   ├── all_models_radar.png
│   ├── all_models_group_binwise_bleu.png
│   └── train_length_distribution.png
└── logs/
    ├── train_transformer_6.log
    └── ...
```

---

## Architecture Notes

### Vanilla Transformer
Standard encoder-decoder Transformer (Vaswani et al. 2017) with sinusoidal
positional encodings and label-smoothed cross-entropy loss.

### Mamba Seq2Seq
- **Encoder**: Stack of MambaBlocks (S6 selective SSM) with sinusoidal PE.
- **Decoder**: MambaBlocks interleaved with multi-head cross-attention to
  encoder memory — enabling the decoder to attend to the full source context.
- The SSM scan is implemented in pure PyTorch (no CUDA kernel required).
  If the `mamba-ssm` package is installed, the fast CUDA kernel is used automatically.

---

## Key Hyperparameters (config.py)

| Parameter       | Value  |
|-----------------|--------|
| Vocabulary      | 8,000 (shared BPE) |
| Max length      | 128 tokens |
| Batch size      | 64     |
| Warmup steps    | 4,000  |
| Label smoothing | 0.1    |
| Early stopping  | patience = 5 epochs |
| Max epochs      | 30     |
| Optimizer       | Adam (β₁=0.9, β₂=0.98) with Noam schedule |




# Fine-tune a single pretrained model
python finetune.py --config mt5_base

# Fine-tune all mT5 variants
python finetune.py --group mt5

# Run everything (scratch + pretrained) end to end
python run_all.py

# Fine-tune only pretrained models, skip plots
python run_all.py --only pretrained --skip_plots

# Mix: all transformers + mt5_base
python run_all.py --only transformer --model mt5_base