# utils

Data preparation utilities.

| Script | Description |
|--------|-------------|
| `text2json.py` | Convert parallel .sa/.hi files to HuggingFace JSON format |
| `txt_to_tsv.py` | Convert prediction .txt files to linked .tsv (with src + ref columns) |
| `resample_7264_5.py` | Resample raw test set to equal sentence counts per bucket |

## Usage

```bash
# Prepare JSON for HuggingFace datasets
python utils/text2json.py \
    --src data/train.sa --tgt data/train.hi --out data/train.json

# Link predictions with source/reference for evaluation
python utils/txt_to_tsv.py \
    --pred_dir predictions/ \
    --src_file data/test_7264_5.sa \
    --ref_file data/test_7264_5.hi

# Resample raw test set to 1452 sentences per bucket (7260 total)
python utils/resample_7264_5.py \
    --src data/test.sa --tgt data/test.hi \
    --out_src data/test_7264_5.sa --out_tgt data/test_7264_5.hi \
    --plot data/resample_plot.png
```
